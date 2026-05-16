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

    use crate::kernels::train::time_mixer::key_prepare::{
        KeyPrepareBackend,
        io::{
            KeyPrepareForwardInputs,
            KeyPrepareForwardPrimitiveInputs,
            KeyPrepareForwardPrimitiveOutput,
        },
        key_prepare_reference,
    };

    impl<B, C> KeyPrepareBackend for Autodiff<B, C>
    where
        B: KeyPrepareBackend,
        C: CheckpointStrategy,
    {
        fn fused_key_prepare(
            inputs: KeyPrepareForwardPrimitiveInputs<Self>,
        ) -> KeyPrepareForwardPrimitiveOutput<Self> {
            let KeyPrepareForwardPrimitiveInputs {
                key,
                key_removal,
                learning_rate,
                key_replacement,
                head_size,
            } = inputs;

            key_prepare_reference(KeyPrepareForwardInputs {
                key: from_float(key),
                key_removal: from_float(key_removal),
                learning_rate: from_float(learning_rate),
                key_replacement: from_float(key_replacement),
                head_size,
            })
            .to_primitive()
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
            TensorMetadata,
            ops::{FloatTensor, FloatTensorOps},
        },
    };
    use burn_cubecl::{
        CubeBackend,
        CubeElement,
        CubeRuntime,
        FloatElement,
        IntElement,
        cubecl::{CubeDim, calculate_cube_count_elemwise, prelude::*},
        element::BoolElement,
        ops::numeric::{empty_device, zeros_client},
        tensor::CubeTensor,
    };

    use crate::kernels::train::{
        layout::assert_linear_readable,
        time_mixer::key_prepare::{
            KeyPrepareBackend,
            io::{KeyPrepareForwardPrimitiveInputs, KeyPrepareForwardPrimitiveOutput},
            kernel::{
                KeyPrepareBackwardInputsLaunch,
                KeyPrepareBackwardOutputsLaunch,
                KeyPrepareForwardInputsLaunch,
                KeyPrepareStackedForwardOutputLaunch,
                key_prepare_backward_partial_kernel,
                key_prepare_stacked_forward_kernel,
            },
        },
    };

    impl<R, F, I, BT, C> KeyPrepareBackend for Autodiff<CubeBackend<R, F, I, BT>, C>
    where
        R: CubeRuntime,
        F: FloatElement + CubeElement,
        I: IntElement,
        BT: BoolElement,
        C: CheckpointStrategy,
    {
        fn fused_key_prepare(
            inputs: KeyPrepareForwardPrimitiveInputs<Self>,
        ) -> KeyPrepareForwardPrimitiveOutput<Self> {
            let [batch_size, context_len, embedded_dim] = inputs.key.shape().dims();
            let stacked_output = fused_key_prepare_stacked::<R, F, I, BT, C>(inputs);
            let shape = [batch_size, context_len, embedded_dim];

            KeyPrepareForwardPrimitiveOutput {
                replacement_key: slice_branch::<R, F, I, BT, C>(stacked_output.clone(), 0, shape),
                removal_key_normalized: slice_branch::<R, F, I, BT, C>(
                    stacked_output.clone(),
                    1,
                    shape,
                ),
                replacement: slice_branch::<R, F, I, BT, C>(stacked_output, 2, shape),
            }
        }
    }

    #[derive(Debug)]
    struct KeyPrepareBackward;

    impl<R, F, I, BT> Backward<CubeBackend<R, F, I, BT>, 4> for KeyPrepareBackward
    where
        R: CubeRuntime,
        F: FloatElement + CubeElement,
        I: IntElement,
        BT: BoolElement,
    {
        type State = KeyPrepareBackwardState<CubeBackend<R, F, I, BT>>;

        fn backward(
            self,
            ops: Ops<Self::State, 4>,
            grads: &mut Gradients,
            _checkpointer: &mut Checkpointer,
        ) {
            let [
                node_key,
                node_key_removal,
                node_learning_rate,
                node_key_replacement,
            ] = ops.parents;
            let output_grad = grads.consume::<CubeBackend<R, F, I, BT>>(&ops.node);
            let grads_out = key_prepare_backward::<R, F, I, BT>(output_grad, ops.state);

            if let Some(node) = node_key {
                grads.register::<CubeBackend<R, F, I, BT>>(node.id, grads_out.key_grad);
            }
            if let Some(node) = node_key_removal {
                grads.register::<CubeBackend<R, F, I, BT>>(node.id, grads_out.key_removal_grad);
            }
            if let Some(node) = node_learning_rate {
                grads.register::<CubeBackend<R, F, I, BT>>(node.id, grads_out.learning_rate_grad);
            }
            if let Some(node) = node_key_replacement {
                grads.register::<CubeBackend<R, F, I, BT>>(node.id, grads_out.key_replacement_grad);
            }
        }
    }

    #[derive(Debug, Clone)]
    struct KeyPrepareBackwardState<B: burn::tensor::backend::Backend> {
        key: FloatTensor<B>,
        key_removal: FloatTensor<B>,
        learning_rate: FloatTensor<B>,
        key_replacement: FloatTensor<B>,
        head_size: usize,
    }

    struct KeyPrepareBackwardOutput<B: burn::tensor::backend::Backend> {
        key_grad: FloatTensor<B>,
        key_removal_grad: FloatTensor<B>,
        learning_rate_grad: FloatTensor<B>,
        key_replacement_grad: FloatTensor<B>,
    }

    fn fused_key_prepare_stacked<R, F, I, BT, C>(
        inputs: KeyPrepareForwardPrimitiveInputs<Autodiff<CubeBackend<R, F, I, BT>, C>>,
    ) -> FloatTensor<Autodiff<CubeBackend<R, F, I, BT>, C>>
    where
        R: CubeRuntime,
        F: FloatElement + CubeElement,
        I: IntElement,
        BT: BoolElement,
        C: CheckpointStrategy,
    {
        let KeyPrepareForwardPrimitiveInputs {
            key,
            key_removal,
            learning_rate,
            key_replacement,
            head_size,
        } = inputs;

        assert_linear_readable("key", &key.primitive);
        assert_linear_readable("key_removal", &key_removal.primitive);
        assert_linear_readable("learning_rate", &learning_rate.primitive);
        assert_linear_readable("key_replacement", &key_replacement.primitive);

        let key_primitive = key.primitive.clone();
        let key_removal_primitive = key_removal.primitive.clone();
        let learning_rate_primitive = learning_rate.primitive.clone();
        let key_replacement_primitive = key_replacement.primitive.clone();
        let parents = [
            key.node.clone(),
            key_removal.node.clone(),
            learning_rate.node.clone(),
            key_replacement.node.clone(),
        ];
        let stacked_output = key_prepare_stacked_forward::<R, F, I, BT>(
            key_primitive.clone(),
            key_removal_primitive.clone(),
            learning_rate_primitive.clone(),
            key_replacement_primitive.clone(),
            head_size,
        );
        let state = KeyPrepareBackwardState {
            key: key_primitive,
            key_removal: key_removal_primitive,
            learning_rate: learning_rate_primitive,
            key_replacement: key_replacement_primitive,
            head_size,
        };

        match KeyPrepareBackward
            .prepare::<C>(parents)
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(state, stacked_output),
            OpsKind::UnTracked(prep) => prep.finish(stacked_output),
        }
    }

    fn key_prepare_stacked_forward<R, F, I, BT>(
        key: CubeTensor<R>,
        key_removal: CubeTensor<R>,
        learning_rate: CubeTensor<R>,
        key_replacement: CubeTensor<R>,
        head_size: usize,
    ) -> FloatTensor<CubeBackend<R, F, I, BT>>
    where
        R: CubeRuntime,
        F: FloatElement + CubeElement,
        I: IntElement,
        BT: BoolElement,
    {
        let shape = key.meta.shape().clone();
        let client = key.client.clone();
        let output_shape = Shape::new([3, shape[0], shape[1], shape[2]]);
        let output = empty_device::<R, F>(client.clone(), key.device.clone(), output_shape);

        if shape.num_elements() == 0 {
            return output;
        }

        let working_units = shape.num_elements();
        let cube_dim = CubeDim::new(&client, working_units);
        let cube_count = calculate_cube_count_elemwise(&client, working_units, cube_dim);
        let address_type = max_address_type(&[
            &key,
            &key_removal,
            &learning_rate,
            &key_replacement,
            &output,
        ]);
        let head_mask = if head_size > 1 && head_size.is_power_of_two() {
            head_size - 1
        } else {
            0
        };

        unsafe {
            key_prepare_stacked_forward_kernel::launch_unchecked::<F, R>(
                &client,
                cube_count,
                cube_dim,
                address_type,
                KeyPrepareForwardInputsLaunch::new(
                    key.into_linear_view(),
                    key_removal.into_linear_view(),
                    learning_rate.into_linear_view(),
                    key_replacement.into_linear_view(),
                ),
                KeyPrepareStackedForwardOutputLaunch::new(output.clone().into_linear_view()),
                shape[2],
                head_size,
                head_mask,
                working_units,
            );
        }

        output
    }

    fn key_prepare_backward<R, F, I, BT>(
        output_grad: CubeTensor<R>,
        state: KeyPrepareBackwardState<CubeBackend<R, F, I, BT>>,
    ) -> KeyPrepareBackwardOutput<CubeBackend<R, F, I, BT>>
    where
        R: CubeRuntime,
        F: FloatElement + CubeElement,
        I: IntElement,
        BT: BoolElement,
    {
        let KeyPrepareBackwardState {
            key,
            key_removal,
            learning_rate,
            key_replacement,
            head_size,
        } = state;
        let shape = key.meta.shape().clone();
        let [_batch_size, _context_len, embedded_dim] = shape.dims();
        let num_elements = shape.num_elements();
        let client = key.client.clone();
        let device = key.device.clone();
        let scale_shape = Shape::new([embedded_dim]);
        let key_grad = empty_device::<R, F>(client.clone(), device.clone(), shape.clone());
        let learning_rate_grad =
            empty_device::<R, F>(client.clone(), device.clone(), shape.clone());
        let key_removal_grad = zeros_client::<R>(
            client.clone(),
            device.clone(),
            scale_shape.clone(),
            key_removal.dtype,
        );
        let key_replacement_grad =
            zeros_client::<R>(client.clone(), device, scale_shape, key_replacement.dtype);
        let cube_dim = CubeDim::new(&client, num_elements);
        let cube_count = calculate_cube_count_elemwise(&client, num_elements, cube_dim);
        let partial_address_type = max_address_type(&[
            &output_grad,
            &key,
            &key_removal,
            &learning_rate,
            &key_replacement,
            &key_grad,
            &key_removal_grad,
            &learning_rate_grad,
            &key_replacement_grad,
        ]);
        let head_mask = if head_size > 1 && head_size.is_power_of_two() {
            head_size - 1
        } else {
            0
        };

        unsafe {
            key_prepare_backward_partial_kernel::launch_unchecked::<F, R>(
                &client,
                cube_count,
                cube_dim,
                partial_address_type,
                KeyPrepareBackwardInputsLaunch::new(
                    output_grad.into_linear_view(),
                    key.into_linear_view_like(&key_grad),
                    key_removal.into_linear_view(),
                    learning_rate.into_linear_view_like(&learning_rate_grad),
                    key_replacement.into_linear_view(),
                ),
                KeyPrepareBackwardOutputsLaunch::new(
                    key_grad.clone().into_linear_view(),
                    key_removal_grad.clone().into_linear_view(),
                    learning_rate_grad.clone().into_linear_view(),
                    key_replacement_grad.clone().into_linear_view(),
                ),
                embedded_dim,
                head_size,
                head_mask,
                num_elements,
            );
        }

        KeyPrepareBackwardOutput {
            key_grad,
            key_removal_grad,
            learning_rate_grad,
            key_replacement_grad,
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

    fn max_address_type<R: CubeRuntime>(tensors: &[&CubeTensor<R>]) -> AddressType {
        tensors
            .iter()
            .map(|tensor| tensor.required_address_type())
            .max()
            .unwrap_or_default()
    }
}
