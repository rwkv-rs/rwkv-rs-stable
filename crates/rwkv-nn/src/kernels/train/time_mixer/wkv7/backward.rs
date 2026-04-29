#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal"
))]
use burn::backend::autodiff::{
    NodeId,
    checkpoint::base::Checkpointer,
    grads::Gradients,
    ops::{Backward, Ops, OpsKind},
};
use burn::{
    backend::autodiff::{Autodiff, checkpoint::strategy::CheckpointStrategy},
    tensor::ops::FloatTensor,
};
#[cfg(not(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal"
)))]
use burn::tensor::{Tensor, TensorPrimitive};
#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal"
))]
use burn::tensor::Shape;
#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal"
))]
use burn_cubecl::{
    CubeBackend,
    CubeElement,
    CubeRuntime,
    FloatElement,
    IntElement,
    cubecl::{CubeCount, CubeDim, prelude::*},
    element::BoolElement,
    ops::numeric::{empty_device, zeros_client},
    tensor::CubeTensor,
};

#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal"
))]
use crate::kernels::train::time_mixer::wkv7::{
    forward,
    kernel::{Wkv7BackwardInputsLaunch, Wkv7BackwardOutputsLaunch, wkv7_backward_kernel},
};
use crate::kernels::train::time_mixer::wkv7::{
    Wkv7Backend,
    io::{
        Wkv7PretrainForwardPrimitiveInputs,
        Wkv7StatepassForwardPrimitiveInputs,
        Wkv7StatepassForwardPrimitiveOutput,
        Wkv7StatetuneForwardPrimitiveInputs,
    },
};
#[cfg(not(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal"
)))]
use crate::kernels::train::time_mixer::wkv7::{
    io::{Wkv7PretrainForwardInputs, Wkv7StatepassForwardInputs, Wkv7StatetuneForwardInputs},
    wkv7_pretrain_reference,
    wkv7_statepass_reference,
    wkv7_statetune_reference,
};

#[cfg(not(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal"
)))]
impl<B, C> Wkv7Backend for Autodiff<B, C>
where
    B: Wkv7Backend,
    C: CheckpointStrategy,
{
    fn fused_wkv7_pretrain(inputs: Wkv7PretrainForwardPrimitiveInputs<Self>) -> FloatTensor<Self> {
        let output = wkv7_pretrain_reference(Wkv7PretrainForwardInputs {
            receptance: from_float::<Self, 4>(inputs.receptance),
            weight_decay: from_float::<Self, 4>(inputs.weight_decay),
            replacement_key: from_float::<Self, 4>(inputs.replacement_key),
            value: from_float::<Self, 4>(inputs.value),
            removal_key_normalized: from_float::<Self, 4>(inputs.removal_key_normalized),
            replacement: from_float::<Self, 4>(inputs.replacement),
            chunk_len: inputs.chunk_len,
        });

        output.into_primitive().tensor()
    }

    fn fused_wkv7_statetune(
        inputs: Wkv7StatetuneForwardPrimitiveInputs<Self>,
    ) -> FloatTensor<Self> {
        let output = wkv7_statetune_reference(Wkv7StatetuneForwardInputs {
            initial_state: from_float::<Self, 4>(inputs.initial_state),
            sequence: Wkv7PretrainForwardInputs {
                receptance: from_float::<Self, 4>(inputs.sequence.receptance),
                weight_decay: from_float::<Self, 4>(inputs.sequence.weight_decay),
                replacement_key: from_float::<Self, 4>(inputs.sequence.replacement_key),
                value: from_float::<Self, 4>(inputs.sequence.value),
                removal_key_normalized: from_float::<Self, 4>(
                    inputs.sequence.removal_key_normalized,
                ),
                replacement: from_float::<Self, 4>(inputs.sequence.replacement),
                chunk_len: inputs.sequence.chunk_len,
            },
        });

        output.into_primitive().tensor()
    }

    fn fused_wkv7_statepass(
        inputs: Wkv7StatepassForwardPrimitiveInputs<Self>,
    ) -> Wkv7StatepassForwardPrimitiveOutput<Self> {
        let output = wkv7_statepass_reference(Wkv7StatepassForwardInputs {
            initial_state: from_float::<Self, 4>(inputs.initial_state),
            sequence: Wkv7PretrainForwardInputs {
                receptance: from_float::<Self, 4>(inputs.sequence.receptance),
                weight_decay: from_float::<Self, 4>(inputs.sequence.weight_decay),
                replacement_key: from_float::<Self, 4>(inputs.sequence.replacement_key),
                value: from_float::<Self, 4>(inputs.sequence.value),
                removal_key_normalized: from_float::<Self, 4>(
                    inputs.sequence.removal_key_normalized,
                ),
                replacement: from_float::<Self, 4>(inputs.sequence.replacement),
                chunk_len: inputs.sequence.chunk_len,
            },
        });

        Wkv7StatepassForwardPrimitiveOutput {
            output: output.output.into_primitive().tensor(),
            next_state: output.next_state.into_primitive().tensor(),
        }
    }
}

#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal"
))]
impl<R, F, I, BT, C> Wkv7Backend for Autodiff<CubeBackend<R, F, I, BT>, C>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
    C: CheckpointStrategy,
{
    fn fused_wkv7_pretrain(inputs: Wkv7PretrainForwardPrimitiveInputs<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Wkv7PretrainBackward;

        impl<R, F, I, BT> Backward<CubeBackend<R, F, I, BT>, 6> for Wkv7PretrainBackward
        where
            R: CubeRuntime,
            F: FloatElement + CubeElement,
            I: IntElement,
            BT: BoolElement,
        {
            type State = Wkv7BackwardState<R>;

            fn backward(
                self,
                ops: Ops<Self::State, 6>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let [
                    node_receptance,
                    node_weight_decay,
                    node_replacement_key,
                    node_value,
                    node_removal_key_normalized,
                    node_replacement,
                ] = ops.parents;
                let output_grad = grads.consume::<CubeBackend<R, F, I, BT>>(&ops.node);
                let state = ops.state;
                let next_state_grad = zero_state_grad::<R>(&state.receptance);
                let grads_out =
                    launch_wkv7_backward::<R, F, I, BT>(state, output_grad, next_state_grad);

                register_sequence_grads::<R, F, I, BT>(
                    grads,
                    [
                        node_receptance.map(|node| node.id),
                        node_weight_decay.map(|node| node.id),
                        node_replacement_key.map(|node| node.id),
                        node_value.map(|node| node.id),
                        node_removal_key_normalized.map(|node| node.id),
                        node_replacement.map(|node| node.id),
                    ],
                    grads_out,
                );
            }
        }

        let Wkv7PretrainForwardPrimitiveInputs {
            receptance,
            weight_decay,
            replacement_key,
            value,
            removal_key_normalized,
            replacement,
            chunk_len,
        } = inputs;
        let output = forward::fused_wkv7_pretrain_with_saved::<R, F, I, BT>(
            Wkv7PretrainForwardPrimitiveInputs {
                receptance: receptance.primitive.clone(),
                weight_decay: weight_decay.primitive.clone(),
                replacement_key: replacement_key.primitive.clone(),
                value: value.primitive.clone(),
                removal_key_normalized: removal_key_normalized.primitive.clone(),
                replacement: replacement.primitive.clone(),
                chunk_len,
            },
        );

        match Wkv7PretrainBackward
            .prepare::<C>([
                receptance.node.clone(),
                weight_decay.node.clone(),
                replacement_key.node.clone(),
                value.node.clone(),
                removal_key_normalized.node.clone(),
                replacement.node.clone(),
            ])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(
                Wkv7BackwardState {
                    receptance: receptance.primitive,
                    weight_decay: weight_decay.primitive,
                    replacement_key: replacement_key.primitive,
                    value: value.primitive,
                    removal_key_normalized: removal_key_normalized.primitive,
                    replacement: replacement.primitive,
                    snapshots: output.snapshots,
                    state_replacement: output.state_replacement,
                    chunk_len,
                },
                output.output,
            ),
            OpsKind::UnTracked(prep) => prep.finish(output.output),
        }
    }

    fn fused_wkv7_statetune(
        inputs: Wkv7StatetuneForwardPrimitiveInputs<Self>,
    ) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct Wkv7StatetuneBackward;

        impl<R, F, I, BT> Backward<CubeBackend<R, F, I, BT>, 7> for Wkv7StatetuneBackward
        where
            R: CubeRuntime,
            F: FloatElement + CubeElement,
            I: IntElement,
            BT: BoolElement,
        {
            type State = Wkv7BackwardState<R>;

            fn backward(
                self,
                ops: Ops<Self::State, 7>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let [
                    node_initial_state,
                    node_receptance,
                    node_weight_decay,
                    node_replacement_key,
                    node_value,
                    node_removal_key_normalized,
                    node_replacement,
                ] = ops.parents;
                let output_grad = grads.consume::<CubeBackend<R, F, I, BT>>(&ops.node);
                let state = ops.state;
                let next_state_grad = zero_state_grad::<R>(&state.receptance);
                let grads_out =
                    launch_wkv7_backward::<R, F, I, BT>(state, output_grad, next_state_grad);

                if let Some(node) = node_initial_state {
                    grads.register::<CubeBackend<R, F, I, BT>>(
                        node.id,
                        grads_out.initial_state_grad.clone(),
                    );
                }
                register_sequence_grads::<R, F, I, BT>(
                    grads,
                    [
                        node_receptance.map(|node| node.id),
                        node_weight_decay.map(|node| node.id),
                        node_replacement_key.map(|node| node.id),
                        node_value.map(|node| node.id),
                        node_removal_key_normalized.map(|node| node.id),
                        node_replacement.map(|node| node.id),
                    ],
                    grads_out,
                );
            }
        }

        let Wkv7StatetuneForwardPrimitiveInputs {
            initial_state,
            sequence,
        } = inputs;
        let Wkv7PretrainForwardPrimitiveInputs {
            receptance,
            weight_decay,
            replacement_key,
            value,
            removal_key_normalized,
            replacement,
            chunk_len,
        } = sequence;
        let output = forward::fused_wkv7_statetune_with_saved::<R, F, I, BT>(
            Wkv7StatetuneForwardPrimitiveInputs {
                initial_state: initial_state.primitive.clone(),
                sequence: Wkv7PretrainForwardPrimitiveInputs {
                    receptance: receptance.primitive.clone(),
                    weight_decay: weight_decay.primitive.clone(),
                    replacement_key: replacement_key.primitive.clone(),
                    value: value.primitive.clone(),
                    removal_key_normalized: removal_key_normalized.primitive.clone(),
                    replacement: replacement.primitive.clone(),
                    chunk_len,
                },
            },
        );

        match Wkv7StatetuneBackward
            .prepare::<C>([
                initial_state.node.clone(),
                receptance.node.clone(),
                weight_decay.node.clone(),
                replacement_key.node.clone(),
                value.node.clone(),
                removal_key_normalized.node.clone(),
                replacement.node.clone(),
            ])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(
                Wkv7BackwardState {
                    receptance: receptance.primitive,
                    weight_decay: weight_decay.primitive,
                    replacement_key: replacement_key.primitive,
                    value: value.primitive,
                    removal_key_normalized: removal_key_normalized.primitive,
                    replacement: replacement.primitive,
                    snapshots: output.snapshots,
                    state_replacement: output.state_replacement,
                    chunk_len,
                },
                output.output,
            ),
            OpsKind::UnTracked(prep) => prep.finish(output.output),
        }
    }

    fn fused_wkv7_statepass(
        inputs: Wkv7StatepassForwardPrimitiveInputs<Self>,
    ) -> Wkv7StatepassForwardPrimitiveOutput<Self> {
        #[derive(Clone, Copy, Debug)]
        enum Wkv7StatepassBackwardTarget {
            Output,
            NextState,
        }

        #[derive(Debug)]
        struct Wkv7StatepassBackward {
            target: Wkv7StatepassBackwardTarget,
        }

        impl<R, F, I, BT> Backward<CubeBackend<R, F, I, BT>, 7> for Wkv7StatepassBackward
        where
            R: CubeRuntime,
            F: FloatElement + CubeElement,
            I: IntElement,
            BT: BoolElement,
        {
            type State = Wkv7BackwardState<R>;

            fn backward(
                self,
                ops: Ops<Self::State, 7>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let [
                    node_initial_state,
                    node_receptance,
                    node_weight_decay,
                    node_replacement_key,
                    node_value,
                    node_removal_key_normalized,
                    node_replacement,
                ] = ops.parents;
                let current_grad = grads.consume::<CubeBackend<R, F, I, BT>>(&ops.node);
                let state = ops.state;
                let (output_grad, next_state_grad) = match self.target {
                    Wkv7StatepassBackwardTarget::Output => {
                        (current_grad, zero_state_grad::<R>(&state.receptance))
                    }
                    Wkv7StatepassBackwardTarget::NextState => {
                        (zero_sequence_grad::<R>(&state.receptance), current_grad)
                    }
                };
                let grads_out =
                    launch_wkv7_backward::<R, F, I, BT>(state, output_grad, next_state_grad);

                if let Some(node) = node_initial_state {
                    grads.register::<CubeBackend<R, F, I, BT>>(
                        node.id,
                        grads_out.initial_state_grad.clone(),
                    );
                }
                register_sequence_grads::<R, F, I, BT>(
                    grads,
                    [
                        node_receptance.map(|node| node.id),
                        node_weight_decay.map(|node| node.id),
                        node_replacement_key.map(|node| node.id),
                        node_value.map(|node| node.id),
                        node_removal_key_normalized.map(|node| node.id),
                        node_replacement.map(|node| node.id),
                    ],
                    grads_out,
                );
            }
        }

        let Wkv7StatepassForwardPrimitiveInputs {
            initial_state,
            sequence,
        } = inputs;
        let Wkv7PretrainForwardPrimitiveInputs {
            receptance,
            weight_decay,
            replacement_key,
            value,
            removal_key_normalized,
            replacement,
            chunk_len,
        } = sequence;
        let output = forward::fused_wkv7_statepass_with_saved::<R, F, I, BT>(
            Wkv7StatepassForwardPrimitiveInputs {
                initial_state: initial_state.primitive.clone(),
                sequence: Wkv7PretrainForwardPrimitiveInputs {
                    receptance: receptance.primitive.clone(),
                    weight_decay: weight_decay.primitive.clone(),
                    replacement_key: replacement_key.primitive.clone(),
                    value: value.primitive.clone(),
                    removal_key_normalized: removal_key_normalized.primitive.clone(),
                    replacement: replacement.primitive.clone(),
                    chunk_len,
                },
            },
        );

        let parents = [
            initial_state.node.clone(),
            receptance.node.clone(),
            weight_decay.node.clone(),
            replacement_key.node.clone(),
            value.node.clone(),
            removal_key_normalized.node.clone(),
            replacement.node.clone(),
        ];
        let state = Wkv7BackwardState {
            receptance: receptance.primitive,
            weight_decay: weight_decay.primitive,
            replacement_key: replacement_key.primitive,
            value: value.primitive,
            removal_key_normalized: removal_key_normalized.primitive,
            replacement: replacement.primitive,
            snapshots: output.snapshots,
            state_replacement: output.state_replacement,
            chunk_len,
        };
        let output_tensor = match (Wkv7StatepassBackward {
            target: Wkv7StatepassBackwardTarget::Output,
        })
        .prepare::<C>(parents.clone())
        .compute_bound()
        .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(state.clone(), output.output),
            OpsKind::UnTracked(prep) => prep.finish(output.output),
        };
        let next_state = match (Wkv7StatepassBackward {
            target: Wkv7StatepassBackwardTarget::NextState,
        })
        .prepare::<C>(parents)
        .compute_bound()
        .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(state, output.next_state),
            OpsKind::UnTracked(prep) => prep.finish(output.next_state),
        };

        Wkv7StatepassForwardPrimitiveOutput {
            output: output_tensor,
            next_state,
        }
    }
}

#[cfg(not(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal"
)))]
fn from_float<B, const D: usize>(tensor: FloatTensor<B>) -> Tensor<B, D>
where
    B: burn::tensor::backend::Backend,
{
    Tensor::from_primitive(TensorPrimitive::Float(tensor))
}

#[derive(Clone, Debug)]
#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal"
))]
struct Wkv7BackwardState<R: CubeRuntime> {
    receptance: CubeTensor<R>,
    weight_decay: CubeTensor<R>,
    replacement_key: CubeTensor<R>,
    value: CubeTensor<R>,
    removal_key_normalized: CubeTensor<R>,
    replacement: CubeTensor<R>,
    snapshots: CubeTensor<R>,
    state_replacement: CubeTensor<R>,
    chunk_len: usize,
}

#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal"
))]
struct Wkv7BackwardPrimitiveOutputs<R: CubeRuntime> {
    receptance_grad: CubeTensor<R>,
    weight_decay_grad: CubeTensor<R>,
    replacement_key_grad: CubeTensor<R>,
    value_grad: CubeTensor<R>,
    removal_key_normalized_grad: CubeTensor<R>,
    replacement_grad: CubeTensor<R>,
    initial_state_grad: CubeTensor<R>,
}

#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal"
))]
fn launch_wkv7_backward<R, F, I, BT>(
    state: Wkv7BackwardState<R>,
    output_grad: CubeTensor<R>,
    next_state_grad: CubeTensor<R>,
) -> Wkv7BackwardPrimitiveOutputs<R>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
{
    let shape = state.receptance.meta.shape().clone();
    let client = state.receptance.client.clone();
    let device = state.receptance.device.clone();
    let dtype = state.receptance.dtype;
    let state_shape = Shape::new([shape[0], shape[2], shape[3], shape[3]]);
    let state_transposed_scratch =
        empty_device::<R, F>(client.clone(), device.clone(), state_shape.clone());
    let state_grad_scratch =
        empty_device::<R, F>(client.clone(), device.clone(), state_shape.clone());
    let state_grad_transposed_scratch =
        empty_device::<R, F>(client.clone(), device.clone(), state_shape.clone());
    let state_replacement_grad_scratch =
        empty_device::<R, F>(client.clone(), device.clone(), state_shape.clone());
    let receptance_grad = empty_device::<R, F>(client.clone(), device.clone(), shape.clone());
    let weight_decay_grad = empty_device::<R, F>(client.clone(), device.clone(), shape.clone());
    let replacement_key_grad = empty_device::<R, F>(client.clone(), device.clone(), shape.clone());
    let value_grad = empty_device::<R, F>(client.clone(), device.clone(), shape.clone());
    let removal_key_normalized_grad =
        empty_device::<R, F>(client.clone(), device.clone(), shape.clone());
    let replacement_grad = empty_device::<R, F>(client.clone(), device.clone(), shape.clone());
    let initial_state_grad = empty_device::<R, F>(client.clone(), device.clone(), state_shape);

    if shape.num_elements() > 0 {
        let cube_dim = CubeDim::new_1d(shape[3] as u32);
        let cube_count = CubeCount::Static(shape[2] as u32, shape[0] as u32, 1);
        let address_type = max_address_type(&[
            &state.receptance,
            &state.weight_decay,
            &state.replacement_key,
            &state.value,
            &state.removal_key_normalized,
            &state.replacement,
            &output_grad,
            &next_state_grad,
            &state.snapshots,
            &state.state_replacement,
            &state_transposed_scratch,
            &state_grad_scratch,
            &state_grad_transposed_scratch,
            &state_replacement_grad_scratch,
            &receptance_grad,
            &weight_decay_grad,
            &replacement_key_grad,
            &value_grad,
            &removal_key_normalized_grad,
            &replacement_grad,
            &initial_state_grad,
        ]);

        // One cube owns one `[batch_size, num_heads]` pair. Units cooperate over `head_size`
        // lanes with shared vectors for the current time step, while each unit keeps one row and
        // one transposed column of the recurrent state gradient live across the reverse scan.
        // SAFETY: Forward contract enforces matching contiguous sequence tensors, chunk alignment,
        // dtype and device. The state tensors are allocated from the checked sequence shape.
        unsafe {
            wkv7_backward_kernel::launch_unchecked::<R>(
                &client,
                cube_count,
                cube_dim,
                address_type,
                Wkv7BackwardInputsLaunch::new(
                    state.receptance.into_linear_view(),
                    state.weight_decay.into_linear_view(),
                    state.replacement_key.into_linear_view(),
                    state.value.into_linear_view(),
                    state.removal_key_normalized.into_linear_view(),
                    state.replacement.into_linear_view(),
                    output_grad.into_linear_view(),
                    next_state_grad.into_linear_view(),
                    state.snapshots.into_linear_view(),
                    state.state_replacement.into_linear_view(),
                    state_transposed_scratch.into_linear_view(),
                    state_grad_scratch.into_linear_view(),
                    state_grad_transposed_scratch.into_linear_view(),
                    state_replacement_grad_scratch.into_linear_view(),
                ),
                Wkv7BackwardOutputsLaunch::new(
                    receptance_grad.clone().into_linear_view(),
                    weight_decay_grad.clone().into_linear_view(),
                    replacement_key_grad.clone().into_linear_view(),
                    value_grad.clone().into_linear_view(),
                    removal_key_normalized_grad.clone().into_linear_view(),
                    replacement_grad.clone().into_linear_view(),
                    initial_state_grad.clone().into_linear_view(),
                ),
                shape[1],
                shape[2],
                shape[3],
                state.chunk_len,
                dtype.into(),
            );
        }
    }

    Wkv7BackwardPrimitiveOutputs {
        receptance_grad,
        weight_decay_grad,
        replacement_key_grad,
        value_grad,
        removal_key_normalized_grad,
        replacement_grad,
        initial_state_grad,
    }
}

#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal"
))]
fn zero_sequence_grad<R: CubeRuntime>(tensor: &CubeTensor<R>) -> CubeTensor<R> {
    zeros_client::<R>(
        tensor.client.clone(),
        tensor.device.clone(),
        tensor.meta.shape().clone(),
        tensor.dtype,
    )
}

#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal"
))]
fn zero_state_grad<R: CubeRuntime>(sequence: &CubeTensor<R>) -> CubeTensor<R> {
    let shape = sequence.meta.shape();
    zeros_client::<R>(
        sequence.client.clone(),
        sequence.device.clone(),
        Shape::new([shape[0], shape[2], shape[3], shape[3]]),
        sequence.dtype,
    )
}

#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal"
))]
fn register_sequence_grads<R, F, I, BT>(
    grads: &mut Gradients,
    nodes: [Option<NodeId>; 6],
    grads_out: Wkv7BackwardPrimitiveOutputs<R>,
) where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
{
    let [
        node_receptance,
        node_weight_decay,
        node_replacement_key,
        node_value,
        node_removal_key_normalized,
        node_replacement,
    ] = nodes;

    if let Some(node) = node_receptance {
        grads.register::<CubeBackend<R, F, I, BT>>(node, grads_out.receptance_grad);
    }
    if let Some(node) = node_weight_decay {
        grads.register::<CubeBackend<R, F, I, BT>>(node, grads_out.weight_decay_grad);
    }
    if let Some(node) = node_replacement_key {
        grads.register::<CubeBackend<R, F, I, BT>>(node, grads_out.replacement_key_grad);
    }
    if let Some(node) = node_value {
        grads.register::<CubeBackend<R, F, I, BT>>(node, grads_out.value_grad);
    }
    if let Some(node) = node_removal_key_normalized {
        grads.register::<CubeBackend<R, F, I, BT>>(node, grads_out.removal_key_normalized_grad);
    }
    if let Some(node) = node_replacement {
        grads.register::<CubeBackend<R, F, I, BT>>(node, grads_out.replacement_grad);
    }
}

#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal"
))]
fn max_address_type<R: CubeRuntime>(tensors: &[&CubeTensor<R>]) -> AddressType {
    tensors
        .iter()
        .map(|tensor| tensor.required_address_type())
        .max()
        .unwrap_or_default()
}
