#[cfg(not(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal",
    feature = "wgpu",
    feature = "webgpu"
)))]
mod fallback {
    use burn::{
        backend::autodiff::{Autodiff, checkpoint::strategy::CheckpointStrategy},
        tensor::{Tensor, TensorPrimitive, ops::FloatTensor},
    };

    use crate::kernels::train::channel_mixer::{
        ChannelMixerBackend,
        channel_mixer_reference,
        io::{ChannelMixerForwardInputs, ChannelMixerForwardPrimitiveInputs},
    };

    impl<B, C> ChannelMixerBackend for Autodiff<B, C>
    where
        B: ChannelMixerBackend,
        C: CheckpointStrategy,
    {
        fn fused_channel_mixer(
            inputs: ChannelMixerForwardPrimitiveInputs<Self>,
        ) -> FloatTensor<Self> {
            let ChannelMixerForwardPrimitiveInputs {
                embedded_context,
                key_scale,
                key_weight,
                value_weight,
            } = inputs;

            channel_mixer_reference(ChannelMixerForwardInputs {
                embedded_context: from_float::<Self, 3>(embedded_context),
                key_scale: from_float::<Self, 1>(key_scale),
                key_weight: from_float::<Self, 2>(key_weight),
                value_weight: from_float::<Self, 2>(value_weight),
            })
            .into_primitive()
            .tensor()
        }
    }

    fn from_float<B, const D: usize>(tensor: FloatTensor<B>) -> Tensor<B, D>
    where
        B: burn::tensor::backend::Backend,
    {
        Tensor::from_primitive(TensorPrimitive::Float(tensor))
    }
}

#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal",
    feature = "wgpu",
    feature = "webgpu"
))]
mod cube_impl {
    use burn::{
        backend::autodiff::{
            Autodiff,
            checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
            grads::Gradients,
            ops::{Backward, Ops, OpsKind},
        },
        tensor::{
            Shape,
            ops::{FloatTensor, FloatTensorOps},
        },
    };
    use burn_cubecl::{
        CubeBackend,
        CubeElement,
        CubeRuntime,
        CubeTuneId,
        FloatElement,
        IntElement,
        cubecl::{
            CubeCount,
            CubeDim,
            calculate_cube_count_elemwise,
            prelude::*,
            tensor_vector_size_parallel,
            tune::{
                AutotuneKey,
                AutotuneOutput,
                LocalTuner,
                Tunable,
                TunableSet,
                TuneGroup,
                anchor,
                local_tuner,
            },
        },
        element::BoolElement,
        ops::numeric::{empty_device, zeros_client},
        tensor::CubeTensor,
    };
    use serde::{Deserialize, Serialize};

    use crate::kernels::train::channel_mixer::{
        ChannelMixerBackend,
        forward,
        io::{ChannelMixerBackwardPrimitiveOutputs, ChannelMixerForwardPrimitiveInputs},
        kernel::{
            channel_mixer_key_scale_reduce_finalize_kernel,
            channel_mixer_mix_backward_partial_kernel,
            channel_mixer_relu_square_backward_from_output_kernel,
        },
    };

    impl<R, F, I, BT, C> ChannelMixerBackend for Autodiff<CubeBackend<R, F, I, BT>, C>
    where
        R: CubeRuntime,
        F: FloatElement + CubeElement,
        I: IntElement,
        BT: BoolElement,
        C: CheckpointStrategy,
    {
        fn fused_channel_mixer(
            inputs: ChannelMixerForwardPrimitiveInputs<Self>,
        ) -> FloatTensor<Self> {
            #[derive(Debug)]
            struct FusedChannelMixerBackward;

            impl<R, F, I, BT> Backward<CubeBackend<R, F, I, BT>, 4> for FusedChannelMixerBackward
            where
                R: CubeRuntime,
                F: FloatElement + CubeElement,
                I: IntElement,
                BT: BoolElement,
            {
                type State = ChannelMixerBackwardState<CubeBackend<R, F, I, BT>>;

                fn backward(
                    self,
                    ops: Ops<Self::State, 4>,
                    grads: &mut Gradients,
                    _checkpointer: &mut Checkpointer,
                ) {
                    let [
                        node_embedded_context,
                        node_key_scale,
                        node_key_weight,
                        node_value_weight,
                    ] = ops.parents;
                    let output_grad = grads.consume::<CubeBackend<R, F, I, BT>>(&ops.node);
                    let ChannelMixerBackwardState {
                        embedded_context,
                        key_input,
                        activated_key,
                        key_scale,
                        key_weight,
                        value_weight,
                    } = ops.state;

                    let grads_out = channel_mixer_backward::<R, F, I, BT>(
                        output_grad,
                        embedded_context,
                        key_input,
                        activated_key,
                        key_scale,
                        key_weight,
                        value_weight,
                    );

                    if let Some(node) = node_embedded_context {
                        grads.register::<CubeBackend<R, F, I, BT>>(
                            node.id,
                            grads_out.embedded_context_grad,
                        );
                    }
                    if let Some(node) = node_key_scale {
                        grads.register::<CubeBackend<R, F, I, BT>>(
                            node.id,
                            grads_out.key_scale_grad,
                        );
                    }
                    if let Some(node) = node_key_weight {
                        grads.register::<CubeBackend<R, F, I, BT>>(
                            node.id,
                            grads_out.key_weight_grad,
                        );
                    }
                    if let Some(node) = node_value_weight {
                        grads.register::<CubeBackend<R, F, I, BT>>(
                            node.id,
                            grads_out.value_weight_grad,
                        );
                    }
                }
            }

            let ChannelMixerForwardPrimitiveInputs {
                embedded_context,
                key_scale,
                key_weight,
                value_weight,
            } = inputs;

            let embedded_context_primitive = embedded_context.primitive.clone();
            let key_scale_primitive = key_scale.primitive.clone();
            let key_weight_primitive = key_weight.primitive.clone();
            let value_weight_primitive = value_weight.primitive.clone();

            let key_input = forward::channel_mixer_mix::<R, F>(
                embedded_context_primitive.clone(),
                key_scale_primitive.clone(),
            );
            let [batch_size, context_len, embedded_dim] = key_input.meta.shape().dims();
            let rows = batch_size * context_len;
            let key_input_flat = CubeBackend::<R, F, I, BT>::float_reshape(
                key_input.clone(),
                Shape::new([rows, embedded_dim]),
            );
            let key_projection = CubeBackend::<R, F, I, BT>::float_matmul(
                key_input_flat,
                CubeBackend::<R, F, I, BT>::float_transpose(key_weight_primitive.clone()),
            );
            let activated_key = forward::channel_mixer_relu_square::<R, F>(key_projection);
            let output = CubeBackend::<R, F, I, BT>::float_matmul(
                activated_key.clone(),
                CubeBackend::<R, F, I, BT>::float_transpose(value_weight_primitive.clone()),
            );
            let output = CubeBackend::<R, F, I, BT>::float_reshape(
                output,
                Shape::new([batch_size, context_len, embedded_dim]),
            );

            match FusedChannelMixerBackward
                .prepare::<C>([
                    embedded_context.node.clone(),
                    key_scale.node.clone(),
                    key_weight.node.clone(),
                    value_weight.node.clone(),
                ])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(prep) => prep.finish(
                    ChannelMixerBackwardState {
                        embedded_context: embedded_context_primitive,
                        key_input,
                        activated_key,
                        key_scale: key_scale_primitive,
                        key_weight: key_weight_primitive,
                        value_weight: value_weight_primitive,
                    },
                    output,
                ),
                OpsKind::UnTracked(prep) => prep.finish(output),
            }
        }
    }

    #[derive(Clone, Debug)]
    struct ChannelMixerBackwardState<B: burn::tensor::backend::Backend> {
        embedded_context: FloatTensor<B>,
        key_input: FloatTensor<B>,
        activated_key: FloatTensor<B>,
        key_scale: FloatTensor<B>,
        key_weight: FloatTensor<B>,
        value_weight: FloatTensor<B>,
    }

    fn channel_mixer_backward<
        R: CubeRuntime,
        F: FloatElement + CubeElement,
        I: IntElement,
        BT: BoolElement,
    >(
        output_grad: FloatTensor<CubeBackend<R, F, I, BT>>,
        embedded_context: FloatTensor<CubeBackend<R, F, I, BT>>,
        key_input: FloatTensor<CubeBackend<R, F, I, BT>>,
        activated_key: FloatTensor<CubeBackend<R, F, I, BT>>,
        key_scale: FloatTensor<CubeBackend<R, F, I, BT>>,
        key_weight: FloatTensor<CubeBackend<R, F, I, BT>>,
        value_weight: FloatTensor<CubeBackend<R, F, I, BT>>,
    ) -> ChannelMixerBackwardPrimitiveOutputs<CubeBackend<R, F, I, BT>> {
        assert!(
            output_grad.is_contiguous(),
            "output_grad must be contiguous"
        );
        assert!(
            embedded_context.is_contiguous(),
            "embedded_context must be contiguous"
        );
        assert!(key_input.is_contiguous(), "key_input must be contiguous");
        assert!(
            activated_key.is_contiguous(),
            "activated_key must be contiguous"
        );
        assert!(key_scale.is_contiguous(), "key_scale must be contiguous");
        assert!(key_weight.is_contiguous(), "key_weight must be contiguous");
        assert!(
            value_weight.is_contiguous(),
            "value_weight must be contiguous"
        );

        let [batch_size, context_len, embedded_dim] = output_grad.meta.shape().dims();
        let [expanded_dim, _] = key_weight.meta.shape().dims();
        let rows = batch_size * context_len;

        if rows == 0 {
            let client = output_grad.client.clone();
            let device = output_grad.device.clone();

            return ChannelMixerBackwardPrimitiveOutputs {
                embedded_context_grad: zeros_client::<R>(
                    client.clone(),
                    device.clone(),
                    Shape::new([batch_size, context_len, embedded_dim]),
                    output_grad.dtype,
                ),
                key_scale_grad: zeros_client::<R>(
                    client.clone(),
                    device.clone(),
                    Shape::new([embedded_dim]),
                    output_grad.dtype,
                ),
                key_weight_grad: zeros_client::<R>(
                    client.clone(),
                    device.clone(),
                    Shape::new([expanded_dim, embedded_dim]),
                    output_grad.dtype,
                ),
                value_weight_grad: zeros_client::<R>(
                    client,
                    device,
                    Shape::new([embedded_dim, expanded_dim]),
                    output_grad.dtype,
                ),
            };
        }

        let output_grad_flat = CubeBackend::<R, F, I, BT>::float_reshape(
            output_grad,
            Shape::new([rows, embedded_dim]),
        );
        let value_weight_grad = CubeBackend::<R, F, I, BT>::float_matmul(
            CubeBackend::<R, F, I, BT>::float_transpose(output_grad_flat.clone()),
            activated_key.clone(),
        );
        let activated_key_grad =
            CubeBackend::<R, F, I, BT>::float_matmul(output_grad_flat, value_weight);
        let key_projection_grad = channel_mixer_relu_square_backward_from_output::<R, F>(
            activated_key_grad,
            activated_key,
        );
        let key_input_flat =
            CubeBackend::<R, F, I, BT>::float_reshape(key_input, Shape::new([rows, embedded_dim]));
        let key_weight_grad = CubeBackend::<R, F, I, BT>::float_matmul(
            CubeBackend::<R, F, I, BT>::float_transpose(key_projection_grad.clone()),
            key_input_flat,
        );
        let mixed_grad_flat =
            CubeBackend::<R, F, I, BT>::float_matmul(key_projection_grad, key_weight);
        let mixed_grad = CubeBackend::<R, F, I, BT>::float_reshape(
            mixed_grad_flat,
            Shape::new([batch_size, context_len, embedded_dim]),
        );
        let mix_grads =
            channel_mixer_mix_backward::<R, F, I, BT>(mixed_grad, embedded_context, key_scale);

        ChannelMixerBackwardPrimitiveOutputs {
            embedded_context_grad: mix_grads.embedded_context_grad,
            key_scale_grad: mix_grads.key_scale_grad,
            key_weight_grad,
            value_weight_grad,
        }
    }

    fn channel_mixer_relu_square_backward_from_output<R, F>(
        activated_key_grad: CubeTensor<R>,
        activated_key: CubeTensor<R>,
    ) -> CubeTensor<R>
    where
        R: CubeRuntime,
        F: FloatElement + CubeElement,
    {
        let shape = activated_key_grad.meta.shape().clone();
        let client = activated_key_grad.client.clone();
        let key_projection_grad = empty_device::<R, F>(
            client.clone(),
            activated_key_grad.device.clone(),
            shape.clone(),
        );

        if shape.num_elements() == 0 {
            return key_projection_grad;
        }

        let vector_size =
            best_line_size(&[&activated_key_grad, &activated_key], shape.num_dims() - 1);
        let working_units = shape.num_elements() / vector_size;
        let cube_dim = CubeDim::new(&client, working_units);
        let cube_count = calculate_cube_count_elemwise(&client, working_units, cube_dim);
        let address_type =
            max_address_type(&[&activated_key_grad, &activated_key, &key_projection_grad]);

        // Work is independent per projected activation element. The CUDA cmix kernel avoids
        // saving the pre-activation tensor by deriving the ReLU-square multiplier from the saved
        // squared activation output.
        // SAFETY: Forward produces contiguous `activated_key`; matmul output is contiguous; the
        // vector width divides the innermost expanded dimension.
        unsafe {
            channel_mixer_relu_square_backward_from_output_kernel::launch_unchecked::<F, R>(
                &client,
                cube_count,
                cube_dim,
                address_type,
                vector_size,
                activated_key_grad.into_linear_view_like(&key_projection_grad),
                activated_key.into_linear_view_like(&key_projection_grad),
                key_projection_grad.clone().into_linear_view(),
            );
        }

        key_projection_grad
    }

    fn channel_mixer_mix_backward<
        R: CubeRuntime,
        F: FloatElement + CubeElement,
        I: IntElement,
        BT: BoolElement,
    >(
        mixed_grad: CubeTensor<R>,
        embedded_context: CubeTensor<R>,
        key_scale: CubeTensor<R>,
    ) -> ChannelMixerMixBackwardOutputs<CubeBackend<R, F, I, BT>> {
        let client = mixed_grad.client.clone();

        let key = |(mixed_grad, embedded_context, key_scale): &(
            CubeTensor<R>,
            CubeTensor<R>,
            CubeTensor<R>,
        )| {
            let shape = mixed_grad.meta.shape();

            ChannelMixerMixBackwardAutotuneKey {
                dtype: mixed_grad.dtype,
                num_elements: anchor(shape.num_elements(), None, Some(1), None),
                embedded_dim: shape[2],
                bt_len: anchor(shape[0] * shape[1], None, Some(1), None),
                max_line_size: max_line_size_many(
                    &[mixed_grad, embedded_context, key_scale],
                    shape.num_dims() - 1,
                ),
            }
        };

        let input_gen =
            |_key: &ChannelMixerMixBackwardAutotuneKey,
             (mixed_grad, embedded_context, key_scale): &(
                CubeTensor<R>,
                CubeTensor<R>,
                CubeTensor<R>,
            )| { (mixed_grad.copy(), embedded_context.copy(), key_scale.copy()) };

        static TUNER: LocalTuner<ChannelMixerMixBackwardAutotuneKey, CubeTuneId> =
            local_tuner!("channel-mixer-mix-backward");

        let tunables = TUNER.init(move || {
            let launch_group =
                TuneGroup::<ChannelMixerMixBackwardAutotuneKey>::new("line_size_reduce_tile", |_| {
                    1
                });
            let mut set = TunableSet::new(key, input_gen);

            for line_size in LINE_SIZE_CANDIDATES {
                for block_size in KEY_SCALE_REDUCE_BLOCK_SIZE_CANDIDATES {
                    for bt_tile in KEY_SCALE_REDUCE_BT_TILE_CANDIDATES {
                        set = set.with(
                            Tunable::new(
                                &format!("line_{line_size}_block_{block_size}_bt_{bt_tile}"),
                                move |(mixed_grad, embedded_context, key_scale): (
                                    CubeTensor<R>,
                                    CubeTensor<R>,
                                    CubeTensor<R>,
                                )| {
                                    Ok::<_, String>({
                                    let shape = mixed_grad.meta.shape().clone();
                                    let [batch_size, context_len, embedded_dim] = shape.dims();
                                    let bt_len = batch_size * context_len;
                                    let client = mixed_grad.client.clone();
                                    let device = mixed_grad.device.clone();
                                    let embedded_context_grad =
                                        empty_device::<R, F>(
                                            client.clone(),
                                            device.clone(),
                                            shape.clone(),
                                        );
                                    let key_scale_grad_shape = Shape::new([embedded_dim]);
                                    let key_scale_grad = if bt_len == 0 {
                                        zeros_client::<R>(
                                            client.clone(),
                                            device,
                                            key_scale_grad_shape,
                                            mixed_grad.dtype,
                                        )
                                    } else {
                                        let channel_vecs = embedded_dim / line_size;
                                        let num_bt_tiles = bt_len.div_ceil(bt_tile);
                                        let partial_shape =
                                            Shape::new([num_bt_tiles, embedded_dim]);
                                        let partial_key_scale_grad = empty_device::<R, F>(
                                            client.clone(),
                                            device.clone(),
                                            partial_shape,
                                        );
                                        let key_scale_grad = empty_device::<R, F>(
                                            client.clone(),
                                            device.clone(),
                                            key_scale_grad_shape.clone(),
                                        );
                                        let cubes_x = channel_vecs
                                            .div_ceil(block_size as usize)
                                            as u32;
                                        let partial_address_type = max_address_type(&[
                                            &mixed_grad,
                                            &embedded_context,
                                            &key_scale,
                                            &embedded_context_grad,
                                            &partial_key_scale_grad,
                                        ]);

                                        // This launch mirrors CUDA cmix v5's strongest property:
                                        // one sequential scan over a `[batch_size, context_len]`
                                        // tile computes both the context gradient and the
                                        // key-scale reduction contribution. The scan reuses the
                                        // previous context value inside the tile and reads the next
                                        // mixed gradient only when the next token is in the same
                                        // sequence.
                                        // SAFETY: Public inputs are contiguous. The tuner only
                                        // enables vector sizes that divide `embedded_dim`, and the
                                        // kernel bounds-checks channel vectors and BT tiles.
                                        unsafe {
                                            channel_mixer_mix_backward_partial_kernel::launch_unchecked::<F, R>(
                                                &client,
                                                CubeCount::Static(cubes_x, num_bt_tiles as u32, 1),
                                                CubeDim::new_1d(block_size),
                                                partial_address_type,
                                                line_size,
                                                mixed_grad.clone().into_linear_view(),
                                                embedded_context.into_linear_view(),
                                                key_scale.into_linear_view(),
                                                embedded_context_grad.clone().into_linear_view(),
                                                partial_key_scale_grad.clone().into_linear_view(),
                                                channel_vecs,
                                                bt_len,
                                                context_len,
                                                bt_tile,
                                            );
                                        }

                                        let finalize_address_type =
                                            max_address_type(&[
                                                &partial_key_scale_grad,
                                                &key_scale_grad,
                                            ]);

                                        // SAFETY: Partial sums are shaped `[num_bt_tiles,
                                        // embedded_dim]`, so each channel vector has exactly one
                                        // partial value per BT tile.
                                        unsafe {
                                            channel_mixer_key_scale_reduce_finalize_kernel::launch_unchecked::<F, R>(
                                                &client,
                                                CubeCount::Static(cubes_x, 1, 1),
                                                CubeDim::new_1d(block_size),
                                                finalize_address_type,
                                                line_size,
                                                partial_key_scale_grad.into_linear_view(),
                                                key_scale_grad.clone().into_linear_view(),
                                                channel_vecs,
                                                num_bt_tiles,
                                            );
                                        }

                                        key_scale_grad
                                    };

                                    ChannelMixerMixBackwardOutputs::<CubeBackend<R, F, I, BT>> {
                                        embedded_context_grad,
                                        key_scale_grad,
                                    }
                                    })
                                },
                            )
                            .group(&launch_group, move |key| {
                                if line_size <= key.max_line_size
                                    && key.embedded_dim.is_multiple_of(line_size)
                                {
                                    1
                                } else {
                                    -1
                                }
                            }),
                        );
                    }
                }
            }

            set
        });

        TUNER.execute(
            &CubeTuneId::new(&mixed_grad.client, &mixed_grad.device),
            &client,
            tunables,
            (mixed_grad, embedded_context, key_scale),
        )
    }

    #[derive(Debug, Clone)]
    struct ChannelMixerMixBackwardOutputs<B: burn::tensor::backend::Backend> {
        embedded_context_grad: FloatTensor<B>,
        key_scale_grad: FloatTensor<B>,
    }

    impl<R, F, I, BT> AutotuneOutput for ChannelMixerMixBackwardOutputs<CubeBackend<R, F, I, BT>>
    where
        R: CubeRuntime,
        F: FloatElement,
        I: IntElement,
        BT: BoolElement,
    {
        #[cfg(feature = "autotune-checks")]
        fn check_equivalence(&self, other: Self) {
            AutotuneOutput::check_equivalence(
                &self.embedded_context_grad,
                other.embedded_context_grad,
            );
            AutotuneOutput::check_equivalence(&self.key_scale_grad, other.key_scale_grad);
        }
    }

    const LINE_SIZE_CANDIDATES: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];
    const KEY_SCALE_REDUCE_BLOCK_SIZE_CANDIDATES: [u32; 3] = [64, 128, 256];
    const KEY_SCALE_REDUCE_BT_TILE_CANDIDATES: [usize; 4] = [16, 32, 64, 128];

    #[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
    struct ChannelMixerMixBackwardAutotuneKey {
        dtype: burn::tensor::DType,
        num_elements: usize,
        embedded_dim: usize,
        bt_len: usize,
        max_line_size: usize,
    }

    impl core::fmt::Display for ChannelMixerMixBackwardAutotuneKey {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            write!(
                f,
                "{:?}:{}:{}:{}:{}",
                self.dtype, self.num_elements, self.embedded_dim, self.bt_len, self.max_line_size
            )
        }
    }

    impl AutotuneKey for ChannelMixerMixBackwardAutotuneKey {}

    fn best_line_size<R: CubeRuntime>(tensors: &[&CubeTensor<R>], axis: usize) -> usize {
        let max_line_size = max_line_size_many(tensors, axis);
        let innermost_dim = tensors[0].meta.shape()[axis];

        for line_size in LINE_SIZE_CANDIDATES.into_iter().rev() {
            if line_size <= max_line_size && innermost_dim.is_multiple_of(line_size) {
                return line_size;
            }
        }

        1
    }

    fn max_line_size_many<R: CubeRuntime>(tensors: &[&CubeTensor<R>], axis: usize) -> usize {
        tensors
            .iter()
            .map(|tensor| {
                let tensor_axis = axis.min(tensor.meta.shape().num_dims().saturating_sub(1));
                tensor_vector_size_parallel(
                    tensor.client.io_optimized_vector_sizes(tensor.dtype.size()),
                    tensor.meta.shape(),
                    tensor.meta.strides(),
                    tensor_axis,
                )
            })
            .min()
            .unwrap_or(1)
            .max(1)
    }

    fn max_address_type<R: CubeRuntime>(tensors: &[&CubeTensor<R>]) -> AddressType {
        tensors
            .iter()
            .map(|tensor| tensor.required_address_type())
            .max()
            .unwrap_or_default()
    }
}
