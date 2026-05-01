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

    use crate::kernels::train::time_mixer::mix6::{
        Mix6Backend,
        io::{Mix6ForwardInputs, Mix6ForwardPrimitiveInputs, Mix6ForwardPrimitiveOutput},
        mix6_reference,
    };

    impl<B, C> Mix6Backend for Autodiff<B, C>
    where
        B: Mix6Backend,
        C: CheckpointStrategy,
    {
        fn fused_mix6(
            inputs: Mix6ForwardPrimitiveInputs<Self>,
        ) -> Mix6ForwardPrimitiveOutput<Self> {
            let Mix6ForwardPrimitiveInputs {
                embedded_context,
                receptance_scale,
                weight_decay_scale,
                key_scale,
                value_scale,
                learning_rate_scale,
                gate_scale,
            } = inputs;

            let output = mix6_reference(Mix6ForwardInputs {
                embedded_context: from_float(embedded_context),
                receptance_scale: from_float(receptance_scale),
                weight_decay_scale: from_float(weight_decay_scale),
                key_scale: from_float(key_scale),
                value_scale: from_float(value_scale),
                learning_rate_scale: from_float(learning_rate_scale),
                gate_scale: from_float(gate_scale),
            });

            output.to_primitive()
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
            Slice,
            Tensor,
            TensorMetadata,
            TensorPrimitive,
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

    use crate::kernels::train::time_mixer::mix6::{
        Mix6Backend,
        io::{Mix6ForwardInputs, Mix6ForwardPrimitiveInputs, Mix6ForwardPrimitiveOutput},
        kernel::{
            Mix6BackwardFinalizeInputsLaunch,
            Mix6BackwardFinalizeOutputsLaunch,
            Mix6BackwardInputsLaunch,
            Mix6BackwardOutputsLaunch,
            Mix6ForwardInputsLaunch,
            Mix6StackedForwardOutputLaunch,
            mix6_backward_finalize_kernel,
            mix6_backward_partial_kernel,
            mix6_stacked_forward_kernel,
        },
        mix6_reference,
    };

    impl<R, F, I, BT, C> Mix6Backend for Autodiff<CubeBackend<R, F, I, BT>, C>
    where
        R: CubeRuntime,
        F: FloatElement + CubeElement,
        I: IntElement,
        BT: BoolElement,
        C: CheckpointStrategy,
    {
        fn fused_mix6(
            inputs: Mix6ForwardPrimitiveInputs<Self>,
        ) -> Mix6ForwardPrimitiveOutput<Self> {
            #[derive(Debug)]
            struct FusedMix6Backward;

            impl<R, F, I, BT> Backward<CubeBackend<R, F, I, BT>, 7> for FusedMix6Backward
            where
                R: CubeRuntime,
                F: FloatElement + CubeElement,
                I: IntElement,
                BT: BoolElement,
            {
                type State = Mix6BackwardState<CubeBackend<R, F, I, BT>>;

                fn backward(
                    self,
                    ops: Ops<Self::State, 7>,
                    grads: &mut Gradients,
                    _checkpointer: &mut Checkpointer,
                ) {
                    let [
                        node_embedded_context,
                        node_receptance_scale,
                        node_weight_decay_scale,
                        node_key_scale,
                        node_value_scale,
                        node_learning_rate_scale,
                        node_gate_scale,
                    ] = ops.parents;
                    let output_grad = grads.consume::<CubeBackend<R, F, I, BT>>(&ops.node);
                    let grads_out = mix6_backward::<R, F, I, BT>(output_grad, ops.state);

                    if let Some(node) = node_embedded_context {
                        grads.register::<CubeBackend<R, F, I, BT>>(
                            node.id,
                            grads_out.embedded_context_grad,
                        );
                    }
                    if let Some(node) = node_receptance_scale {
                        grads.register::<CubeBackend<R, F, I, BT>>(
                            node.id,
                            grads_out.receptance_scale_grad,
                        );
                    }
                    if let Some(node) = node_weight_decay_scale {
                        grads.register::<CubeBackend<R, F, I, BT>>(
                            node.id,
                            grads_out.weight_decay_scale_grad,
                        );
                    }
                    if let Some(node) = node_key_scale {
                        grads.register::<CubeBackend<R, F, I, BT>>(
                            node.id,
                            grads_out.key_scale_grad,
                        );
                    }
                    if let Some(node) = node_value_scale {
                        grads.register::<CubeBackend<R, F, I, BT>>(
                            node.id,
                            grads_out.value_scale_grad,
                        );
                    }
                    if let Some(node) = node_learning_rate_scale {
                        grads.register::<CubeBackend<R, F, I, BT>>(
                            node.id,
                            grads_out.learning_rate_scale_grad,
                        );
                    }
                    if let Some(node) = node_gate_scale {
                        grads.register::<CubeBackend<R, F, I, BT>>(
                            node.id,
                            grads_out.gate_scale_grad,
                        );
                    }
                }
            }

            let Mix6ForwardPrimitiveInputs {
                embedded_context,
                receptance_scale,
                weight_decay_scale,
                key_scale,
                value_scale,
                learning_rate_scale,
                gate_scale,
            } = inputs;

            let [batch_size, context_len, embedded_dim] = embedded_context.shape().dims();
            if embedded_context.shape().num_elements() == 0 {
                return mix6_reference(Mix6ForwardInputs {
                    embedded_context: from_float(embedded_context),
                    receptance_scale: from_float(receptance_scale),
                    weight_decay_scale: from_float(weight_decay_scale),
                    key_scale: from_float(key_scale),
                    value_scale: from_float(value_scale),
                    learning_rate_scale: from_float(learning_rate_scale),
                    gate_scale: from_float(gate_scale),
                })
                .to_primitive();
            }

            assert!(
                embedded_context.primitive.is_contiguous(),
                "embedded_context must be contiguous"
            );
            assert!(
                receptance_scale.primitive.is_contiguous(),
                "receptance_scale must be contiguous"
            );
            assert!(
                weight_decay_scale.primitive.is_contiguous(),
                "weight_decay_scale must be contiguous"
            );
            assert!(
                key_scale.primitive.is_contiguous(),
                "key_scale must be contiguous"
            );
            assert!(
                value_scale.primitive.is_contiguous(),
                "value_scale must be contiguous"
            );
            assert!(
                learning_rate_scale.primitive.is_contiguous(),
                "learning_rate_scale must be contiguous"
            );
            assert!(
                gate_scale.primitive.is_contiguous(),
                "gate_scale must be contiguous"
            );

            let parents = [
                embedded_context.node.clone(),
                receptance_scale.node.clone(),
                weight_decay_scale.node.clone(),
                key_scale.node.clone(),
                value_scale.node.clone(),
                learning_rate_scale.node.clone(),
                gate_scale.node.clone(),
            ];
            let stacked_output = mix6_stacked_forward::<R, F, I, BT>(
                embedded_context.primitive.clone(),
                receptance_scale.primitive.clone(),
                weight_decay_scale.primitive.clone(),
                key_scale.primitive.clone(),
                value_scale.primitive.clone(),
                learning_rate_scale.primitive.clone(),
                gate_scale.primitive.clone(),
            );
            let state = Mix6BackwardState {
                embedded_context: embedded_context.primitive,
                receptance_scale: receptance_scale.primitive,
                weight_decay_scale: weight_decay_scale.primitive,
                key_scale: key_scale.primitive,
                value_scale: value_scale.primitive,
                learning_rate_scale: learning_rate_scale.primitive,
                gate_scale: gate_scale.primitive,
            };

            let stacked_output = match FusedMix6Backward
                .prepare::<C>(parents)
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(prep) => prep.finish(state, stacked_output),
                OpsKind::UnTracked(prep) => prep.finish(stacked_output),
            };
            let shape = [batch_size, context_len, embedded_dim];

            Mix6ForwardPrimitiveOutput {
                receptance_input: slice_branch::<R, F, I, BT, C>(stacked_output.clone(), 0, shape),
                weight_decay_input: slice_branch::<R, F, I, BT, C>(
                    stacked_output.clone(),
                    1,
                    shape,
                ),
                key_input: slice_branch::<R, F, I, BT, C>(stacked_output.clone(), 2, shape),
                value_input: slice_branch::<R, F, I, BT, C>(stacked_output.clone(), 3, shape),
                learning_rate_input: slice_branch::<R, F, I, BT, C>(
                    stacked_output.clone(),
                    4,
                    shape,
                ),
                gate_input: slice_branch::<R, F, I, BT, C>(stacked_output, 5, shape),
            }
        }
    }

    #[derive(Debug, Clone)]
    struct Mix6BackwardState<B: burn::tensor::backend::Backend> {
        embedded_context: FloatTensor<B>,
        receptance_scale: FloatTensor<B>,
        weight_decay_scale: FloatTensor<B>,
        key_scale: FloatTensor<B>,
        value_scale: FloatTensor<B>,
        learning_rate_scale: FloatTensor<B>,
        gate_scale: FloatTensor<B>,
    }

    #[derive(Debug, Clone)]
    struct Mix6BackwardPrimitiveOutput<B: burn::tensor::backend::Backend> {
        embedded_context_grad: FloatTensor<B>,
        receptance_scale_grad: FloatTensor<B>,
        weight_decay_scale_grad: FloatTensor<B>,
        key_scale_grad: FloatTensor<B>,
        value_scale_grad: FloatTensor<B>,
        learning_rate_scale_grad: FloatTensor<B>,
        gate_scale_grad: FloatTensor<B>,
    }

    impl<R, F, I, BT> AutotuneOutput for Mix6BackwardPrimitiveOutput<CubeBackend<R, F, I, BT>>
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
            AutotuneOutput::check_equivalence(
                &self.receptance_scale_grad,
                other.receptance_scale_grad,
            );
            AutotuneOutput::check_equivalence(
                &self.weight_decay_scale_grad,
                other.weight_decay_scale_grad,
            );
            AutotuneOutput::check_equivalence(&self.key_scale_grad, other.key_scale_grad);
            AutotuneOutput::check_equivalence(&self.value_scale_grad, other.value_scale_grad);
            AutotuneOutput::check_equivalence(
                &self.learning_rate_scale_grad,
                other.learning_rate_scale_grad,
            );
            AutotuneOutput::check_equivalence(&self.gate_scale_grad, other.gate_scale_grad);
        }
    }

    fn slice_branch<R, F, I, BT, C>(
        stacked_output: FloatTensor<Autodiff<CubeBackend<R, F, I, BT>, C>>,
        branch_index: usize,
        shape: [usize; 3],
    ) -> FloatTensor<Autodiff<CubeBackend<R, F, I, BT>, C>>
    where
        R: CubeRuntime,
        F: FloatElement + CubeElement,
        I: IntElement,
        BT: BoolElement,
        C: CheckpointStrategy,
    {
        let start = branch_index as isize;
        let output = Autodiff::<CubeBackend<R, F, I, BT>, C>::float_slice(
            stacked_output,
            &[
                Slice::new(start, Some(start + 1), 1),
                Slice::full(),
                Slice::full(),
                Slice::full(),
            ],
        );

        Autodiff::<CubeBackend<R, F, I, BT>, C>::float_reshape(output, Shape::new(shape))
    }

    fn mix6_stacked_forward<
        R: CubeRuntime,
        F: FloatElement + CubeElement,
        I: IntElement,
        BT: BoolElement,
    >(
        embedded_context: CubeTensor<R>,
        receptance_scale: CubeTensor<R>,
        weight_decay_scale: CubeTensor<R>,
        key_scale: CubeTensor<R>,
        value_scale: CubeTensor<R>,
        learning_rate_scale: CubeTensor<R>,
        gate_scale: CubeTensor<R>,
    ) -> FloatTensor<CubeBackend<R, F, I, BT>> {
        let shape = embedded_context.meta.shape().clone();
        let client = embedded_context.client.clone();
        let output_shape = Shape::new([6, shape[0], shape[1], shape[2]]);
        let output = empty_device::<R, F>(
            client.clone(),
            embedded_context.device.clone(),
            output_shape,
        );
        let vector_size = best_line_size(
            &[
                &embedded_context,
                &receptance_scale,
                &weight_decay_scale,
                &key_scale,
                &value_scale,
                &learning_rate_scale,
                &gate_scale,
                &output,
            ],
            shape.num_dims() - 1,
        );
        let working_units = shape.num_elements() / vector_size;
        let cube_dim = CubeDim::new(&client, working_units);
        let cube_count = calculate_cube_count_elemwise(&client, working_units, cube_dim);
        let address_type = max_address_type(&[
            &embedded_context,
            &receptance_scale,
            &weight_decay_scale,
            &key_scale,
            &value_scale,
            &learning_rate_scale,
            &gate_scale,
            &output,
        ]);

        // Each work unit computes one embedded-dimension vector for one token and writes all six
        // branch outputs into contiguous branch segments of a stacked tensor. The stacked shape
        // gives Burn autodiff one output node, so the backward pass can consume all branch grads in
        // a single fused scan.
        // SAFETY: The public contract and autodiff entrypoint check shape/dtype/device/contiguity.
        // `best_line_size` only selects vector widths that divide `embedded_dim`.
        unsafe {
            mix6_stacked_forward_kernel::launch_unchecked::<F, R>(
                &client,
                cube_count,
                cube_dim,
                address_type,
                vector_size,
                Mix6ForwardInputsLaunch::new(
                    embedded_context.into_linear_view(),
                    receptance_scale.into_linear_view(),
                    weight_decay_scale.into_linear_view(),
                    key_scale.into_linear_view(),
                    value_scale.into_linear_view(),
                    learning_rate_scale.into_linear_view(),
                    gate_scale.into_linear_view(),
                ),
                Mix6StackedForwardOutputLaunch::new(output.clone().into_linear_view()),
                shape[1],
                working_units,
            );
        }

        output
    }

    fn mix6_backward<
        R: CubeRuntime,
        F: FloatElement + CubeElement,
        I: IntElement,
        BT: BoolElement,
    >(
        output_grad: FloatTensor<CubeBackend<R, F, I, BT>>,
        state: Mix6BackwardState<CubeBackend<R, F, I, BT>>,
    ) -> Mix6BackwardPrimitiveOutput<CubeBackend<R, F, I, BT>> {
        let Mix6BackwardState {
            embedded_context,
            receptance_scale,
            weight_decay_scale,
            key_scale,
            value_scale,
            learning_rate_scale,
            gate_scale,
        } = state;

        assert!(
            output_grad.is_contiguous(),
            "output_grad must be contiguous"
        );
        assert!(
            embedded_context.is_contiguous(),
            "embedded_context must be contiguous"
        );
        assert!(
            receptance_scale.is_contiguous(),
            "receptance_scale must be contiguous"
        );
        assert!(
            weight_decay_scale.is_contiguous(),
            "weight_decay_scale must be contiguous"
        );
        assert!(key_scale.is_contiguous(), "key_scale must be contiguous");
        assert!(
            value_scale.is_contiguous(),
            "value_scale must be contiguous"
        );
        assert!(
            learning_rate_scale.is_contiguous(),
            "learning_rate_scale must be contiguous"
        );
        assert!(gate_scale.is_contiguous(), "gate_scale must be contiguous");

        let client = embedded_context.client.clone();
        let key = |(
            output_grad,
            embedded_context,
            receptance_scale,
            weight_decay_scale,
            key_scale,
            value_scale,
            learning_rate_scale,
            gate_scale,
        ): &(
            CubeTensor<R>,
            CubeTensor<R>,
            CubeTensor<R>,
            CubeTensor<R>,
            CubeTensor<R>,
            CubeTensor<R>,
            CubeTensor<R>,
            CubeTensor<R>,
        )| {
            let shape = embedded_context.meta.shape();

            Mix6BackwardAutotuneKey {
                dtype: embedded_context.dtype,
                num_elements: anchor(shape.num_elements(), None, Some(1), None),
                embedded_dim: shape[2],
                bt_len: anchor(shape[0] * shape[1], None, Some(1), None),
                max_line_size: max_line_size_backward(
                    output_grad,
                    embedded_context,
                    &[
                        receptance_scale,
                        weight_decay_scale,
                        key_scale,
                        value_scale,
                        learning_rate_scale,
                        gate_scale,
                    ],
                ),
            }
        };

        let input_gen = |_key: &Mix6BackwardAutotuneKey,
                         (
            output_grad,
            embedded_context,
            receptance_scale,
            weight_decay_scale,
            key_scale,
            value_scale,
            learning_rate_scale,
            gate_scale,
        ): &(
            CubeTensor<R>,
            CubeTensor<R>,
            CubeTensor<R>,
            CubeTensor<R>,
            CubeTensor<R>,
            CubeTensor<R>,
            CubeTensor<R>,
            CubeTensor<R>,
        )| {
            (
                output_grad.copy(),
                embedded_context.copy(),
                receptance_scale.copy(),
                weight_decay_scale.copy(),
                key_scale.copy(),
                value_scale.copy(),
                learning_rate_scale.copy(),
                gate_scale.copy(),
            )
        };

        static TUNER: LocalTuner<Mix6BackwardAutotuneKey, CubeTuneId> =
            local_tuner!("rwkv7-time-mixer-mix6-backward");

        let tunables =
            TUNER.init(move || {
                let launch_group =
                    TuneGroup::<Mix6BackwardAutotuneKey>::new("line_size_reduce_tile", |_| 1);
                let mut set = TunableSet::new(key, input_gen);

                for line_size in LINE_SIZE_CANDIDATES {
                    for block_size in REDUCE_BLOCK_SIZE_CANDIDATES {
                        for bt_tile in BT_TILE_CANDIDATES {
                            set = set.with(
                            Tunable::new(
                                &format!("line_{line_size}_block_{block_size}_bt_{bt_tile}"),
                                move |(
                                    output_grad,
                                    embedded_context,
                                    receptance_scale,
                                    weight_decay_scale,
                                    key_scale,
                                    value_scale,
                                    learning_rate_scale,
                                    gate_scale,
                                ): (
                                    CubeTensor<R>,
                                    CubeTensor<R>,
                                    CubeTensor<R>,
                                    CubeTensor<R>,
                                    CubeTensor<R>,
                                    CubeTensor<R>,
                                    CubeTensor<R>,
                                    CubeTensor<R>,
                                )| {
                                    Ok::<_, String>({
                                    let shape = embedded_context.meta.shape().clone();
                                    let [batch_size, context_len, embedded_dim] = shape.dims();
                                    let bt_len = batch_size * context_len;
                                    let client = embedded_context.client.clone();
                                    let device = embedded_context.device.clone();
                                    let scale_shape = Shape::new([1, 1, embedded_dim]);

                                    if bt_len == 0 {
                                        return Mix6BackwardPrimitiveOutput::<
                                            CubeBackend<R, F, I, BT>,
                                        > {
                                            embedded_context_grad: zeros_client::<R>(
                                                client.clone(),
                                                device.clone(),
                                                shape,
                                                embedded_context.dtype,
                                            ),
                                            receptance_scale_grad: zeros_client::<R>(
                                                client.clone(),
                                                device.clone(),
                                                scale_shape.clone(),
                                                embedded_context.dtype,
                                            ),
                                            weight_decay_scale_grad: zeros_client::<R>(
                                                client.clone(),
                                                device.clone(),
                                                scale_shape.clone(),
                                                embedded_context.dtype,
                                            ),
                                            key_scale_grad: zeros_client::<R>(
                                                client.clone(),
                                                device.clone(),
                                                scale_shape.clone(),
                                                embedded_context.dtype,
                                            ),
                                            value_scale_grad: zeros_client::<R>(
                                                client.clone(),
                                                device.clone(),
                                                scale_shape.clone(),
                                                embedded_context.dtype,
                                            ),
                                            learning_rate_scale_grad: zeros_client::<R>(
                                                client.clone(),
                                                device.clone(),
                                                scale_shape.clone(),
                                                embedded_context.dtype,
                                            ),
                                            gate_scale_grad: zeros_client::<R>(
                                                client,
                                                device,
                                                scale_shape,
                                                embedded_context.dtype,
                                            ),
                                        };
                                    }

                                    let embedded_context_grad =
                                        empty_device::<R, F>(
                                            client.clone(),
                                            device.clone(),
                                            shape,
                                        );
                                    let num_bt_tiles = bt_len.div_ceil(bt_tile);
                                    let partial_shape = Shape::new([num_bt_tiles, embedded_dim]);
                                    let partial_receptance_scale_grad = empty_device::<R, F>(
                                        client.clone(),
                                        device.clone(),
                                        partial_shape.clone(),
                                    );
                                    let partial_weight_decay_scale_grad = empty_device::<R, F>(
                                        client.clone(),
                                        device.clone(),
                                        partial_shape.clone(),
                                    );
                                    let partial_key_scale_grad = empty_device::<R, F>(
                                        client.clone(),
                                        device.clone(),
                                        partial_shape.clone(),
                                    );
                                    let partial_value_scale_grad = empty_device::<R, F>(
                                        client.clone(),
                                        device.clone(),
                                        partial_shape.clone(),
                                    );
                                    let partial_learning_rate_scale_grad = empty_device::<R, F>(
                                        client.clone(),
                                        device.clone(),
                                        partial_shape.clone(),
                                    );
                                    let partial_gate_scale_grad = empty_device::<R, F>(
                                        client.clone(),
                                        device.clone(),
                                        partial_shape,
                                    );
                                    let receptance_scale_grad = empty_device::<R, F>(
                                        client.clone(),
                                        device.clone(),
                                        scale_shape.clone(),
                                    );
                                    let weight_decay_scale_grad = empty_device::<R, F>(
                                        client.clone(),
                                        device.clone(),
                                        scale_shape.clone(),
                                    );
                                    let key_scale_grad = empty_device::<R, F>(
                                        client.clone(),
                                        device.clone(),
                                        scale_shape.clone(),
                                    );
                                    let value_scale_grad = empty_device::<R, F>(
                                        client.clone(),
                                        device.clone(),
                                        scale_shape.clone(),
                                    );
                                    let learning_rate_scale_grad = empty_device::<R, F>(
                                        client.clone(),
                                        device.clone(),
                                        scale_shape.clone(),
                                    );
                                    let gate_scale_grad = empty_device::<R, F>(
                                        client.clone(),
                                        device.clone(),
                                        scale_shape,
                                    );
                                    let channel_vecs = embedded_dim / line_size;
                                    let cubes_x =
                                        channel_vecs.div_ceil(block_size as usize) as u32;
                                    let partial_address_type = max_address_type(&[
                                        &output_grad,
                                        &embedded_context,
                                        &receptance_scale,
                                        &weight_decay_scale,
                                        &key_scale,
                                        &value_scale,
                                        &learning_rate_scale,
                                        &gate_scale,
                                        &embedded_context_grad,
                                        &partial_receptance_scale_grad,
                                        &partial_weight_decay_scale_grad,
                                        &partial_key_scale_grad,
                                        &partial_value_scale_grad,
                                        &partial_learning_rate_scale_grad,
                                        &partial_gate_scale_grad,
                                    ]);

                                    // One worker streams a `[batch_size, context_len]` tile for one
                                    // embedded-dimension vector. It reuses the six scale vectors,
                                    // reads next-token branch grads only inside a sequence, writes
                                    // `embedded_context_grad`, and accumulates six scale partials.
                                    // SAFETY: The public contract checks input shape/dtype/device,
                                    // the autodiff entrypoint checks contiguity, and autotune only
                                    // enables vector widths that divide `embedded_dim`.
                                    unsafe {
                                        mix6_backward_partial_kernel::launch_unchecked::<F, R>(
                                            &client,
                                            CubeCount::Static(
                                                cubes_x,
                                                num_bt_tiles as u32,
                                                1,
                                            ),
                                            CubeDim::new_1d(block_size),
                                            partial_address_type,
                                            line_size,
                                            Mix6BackwardInputsLaunch::new(
                                                output_grad.into_linear_view(),
                                                embedded_context.into_linear_view(),
                                                receptance_scale.into_linear_view(),
                                                weight_decay_scale.into_linear_view(),
                                                key_scale.into_linear_view(),
                                                value_scale.into_linear_view(),
                                                learning_rate_scale.into_linear_view(),
                                                gate_scale.into_linear_view(),
                                            ),
                                            Mix6BackwardOutputsLaunch::new(
                                                embedded_context_grad
                                                    .clone()
                                                    .into_linear_view(),
                                                partial_receptance_scale_grad
                                                    .clone()
                                                    .into_linear_view(),
                                                partial_weight_decay_scale_grad
                                                    .clone()
                                                    .into_linear_view(),
                                                partial_key_scale_grad
                                                    .clone()
                                                    .into_linear_view(),
                                                partial_value_scale_grad
                                                    .clone()
                                                    .into_linear_view(),
                                                partial_learning_rate_scale_grad
                                                    .clone()
                                                    .into_linear_view(),
                                                partial_gate_scale_grad
                                                    .clone()
                                                    .into_linear_view(),
                                            ),
                                            channel_vecs,
                                            bt_len,
                                            context_len,
                                            bt_tile,
                                        );
                                    }

                                    let finalize_address_type = max_address_type(&[
                                        &partial_receptance_scale_grad,
                                        &partial_weight_decay_scale_grad,
                                        &partial_key_scale_grad,
                                        &partial_value_scale_grad,
                                        &partial_learning_rate_scale_grad,
                                        &partial_gate_scale_grad,
                                        &receptance_scale_grad,
                                        &weight_decay_scale_grad,
                                        &key_scale_grad,
                                        &value_scale_grad,
                                        &learning_rate_scale_grad,
                                        &gate_scale_grad,
                                    ]);

                                    // SAFETY: Partial buffers are shaped `[num_bt_tiles,
                                    // embedded_dim]`, and the finalize kernel bounds-checks the
                                    // tuned channel-vector grid.
                                    unsafe {
                                        mix6_backward_finalize_kernel::launch_unchecked::<F, R>(
                                            &client,
                                            CubeCount::Static(cubes_x, 1, 1),
                                            CubeDim::new_1d(block_size),
                                            finalize_address_type,
                                            line_size,
                                            Mix6BackwardFinalizeInputsLaunch::new(
                                                partial_receptance_scale_grad.into_linear_view(),
                                                partial_weight_decay_scale_grad.into_linear_view(),
                                                partial_key_scale_grad.into_linear_view(),
                                                partial_value_scale_grad.into_linear_view(),
                                                partial_learning_rate_scale_grad.into_linear_view(),
                                                partial_gate_scale_grad.into_linear_view(),
                                            ),
                                            Mix6BackwardFinalizeOutputsLaunch::new(
                                                receptance_scale_grad.clone().into_linear_view(),
                                                weight_decay_scale_grad.clone().into_linear_view(),
                                                key_scale_grad.clone().into_linear_view(),
                                                value_scale_grad.clone().into_linear_view(),
                                                learning_rate_scale_grad
                                                    .clone()
                                                    .into_linear_view(),
                                                gate_scale_grad.clone().into_linear_view(),
                                            ),
                                            channel_vecs,
                                            num_bt_tiles,
                                        );
                                    }

                                    Mix6BackwardPrimitiveOutput::<CubeBackend<R, F, I, BT>> {
                                        embedded_context_grad,
                                        receptance_scale_grad,
                                        weight_decay_scale_grad,
                                        key_scale_grad,
                                        value_scale_grad,
                                        learning_rate_scale_grad,
                                        gate_scale_grad,
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
            &CubeTuneId::new(&embedded_context.client, &embedded_context.device),
            &client,
            tunables,
            (
                output_grad,
                embedded_context,
                receptance_scale,
                weight_decay_scale,
                key_scale,
                value_scale,
                learning_rate_scale,
                gate_scale,
            ),
        )
    }

    const LINE_SIZE_CANDIDATES: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];
    const REDUCE_BLOCK_SIZE_CANDIDATES: [u32; 3] = [64, 128, 256];
    const BT_TILE_CANDIDATES: [usize; 4] = [8, 16, 32, 64];

    #[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
    struct Mix6BackwardAutotuneKey {
        dtype: burn::tensor::DType,
        num_elements: usize,
        embedded_dim: usize,
        bt_len: usize,
        max_line_size: usize,
    }

    impl core::fmt::Display for Mix6BackwardAutotuneKey {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            write!(
                f,
                "{:?}:{}:{}:{}:{}",
                self.dtype, self.num_elements, self.embedded_dim, self.bt_len, self.max_line_size
            )
        }
    }

    impl AutotuneKey for Mix6BackwardAutotuneKey {}

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

    fn max_line_size_backward<R: CubeRuntime>(
        output_grad: &CubeTensor<R>,
        embedded_context: &CubeTensor<R>,
        scales: &[&CubeTensor<R>],
    ) -> usize {
        let output_grad_line_size = tensor_vector_size_parallel(
            output_grad
                .client
                .io_optimized_vector_sizes(output_grad.dtype.size()),
            output_grad.meta.shape(),
            output_grad.meta.strides(),
            output_grad.meta.shape().num_dims() - 1,
        );
        let embedded_context_line_size = tensor_vector_size_parallel(
            embedded_context
                .client
                .io_optimized_vector_sizes(embedded_context.dtype.size()),
            embedded_context.meta.shape(),
            embedded_context.meta.strides(),
            embedded_context.meta.shape().num_dims() - 1,
        );
        let scale_line_size = scales
            .iter()
            .map(|scale| {
                tensor_vector_size_parallel(
                    scale.client.io_optimized_vector_sizes(scale.dtype.size()),
                    scale.meta.shape(),
                    scale.meta.strides(),
                    scale.meta.shape().num_dims() - 1,
                )
            })
            .min()
            .unwrap_or(1);

        output_grad_line_size
            .min(embedded_context_line_size)
            .min(scale_line_size)
            .max(1)
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

    fn from_float<B, const D: usize>(tensor: FloatTensor<B>) -> Tensor<B, D>
    where
        B: burn::tensor::backend::Backend,
    {
        Tensor::from_primitive(TensorPrimitive::Float(tensor))
    }
}
