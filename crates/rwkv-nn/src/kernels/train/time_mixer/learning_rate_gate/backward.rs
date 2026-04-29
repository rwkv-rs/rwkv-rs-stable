use burn::{
    backend::autodiff::{
        Autodiff,
        NodeId,
        checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
        grads::Gradients,
        ops::{Backward, Ops, OpsKind},
    },
    tensor::{Shape, ops::FloatTensor},
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
        prelude::*,
        tensor_vector_size_parallel,
        tune::{
            AsFunctionTunable,
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

use crate::kernels::train::time_mixer::learning_rate_gate::{
    LearningRateGateBackend,
    io::{LearningRateGateBackwardPrimitiveOutputs, LearningRateGateForwardPrimitiveInputs},
    kernel::{
        learning_rate_gate_backward_finalize_kernel,
        learning_rate_gate_backward_partial_kernel,
    },
};

impl<R, F, I, BT, C> LearningRateGateBackend for Autodiff<CubeBackend<R, F, I, BT>, C>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
    C: CheckpointStrategy,
{
    fn fused_learning_rate_gate(
        inputs: LearningRateGateForwardPrimitiveInputs<Self>,
    ) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct LearningRateGateBackward;

        impl<R, F, I, BT> Backward<CubeBackend<R, F, I, BT>, 2> for LearningRateGateBackward
        where
            R: CubeRuntime,
            F: FloatElement + CubeElement,
            I: IntElement,
            BT: BoolElement,
        {
            type State = (NodeId, FloatTensor<CubeBackend<R, F, I, BT>>);

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_learning_rate_base, node_learning_rate_input] = ops.parents;
                let output_grad = grads.consume::<CubeBackend<R, F, I, BT>>(&ops.node);
                let (learning_rate_input_state, learning_rate_base) = ops.state;
                let learning_rate_input: FloatTensor<CubeBackend<R, F, I, BT>> =
                    checkpointer.retrieve_node_output(learning_rate_input_state);

                assert!(
                    learning_rate_base.is_contiguous(),
                    "learning_rate_base must be contiguous"
                );
                assert!(
                    learning_rate_input.is_contiguous(),
                    "learning_rate_input must be contiguous"
                );
                assert!(
                    output_grad.is_contiguous(),
                    "output_grad must be contiguous"
                );

                let client = learning_rate_input.client.clone();
                let key = |learning_rate_base: &CubeTensor<R>,
                           learning_rate_input: &CubeTensor<R>,
                           output_grad: &CubeTensor<R>| {
                    let shape = learning_rate_input.meta.shape();
                    let max_line_size = max_line_size_backward(
                        learning_rate_base,
                        learning_rate_input,
                        output_grad,
                    );

                    LearningRateGateBackwardAutotuneKey {
                        dtype: learning_rate_input.dtype,
                        num_elements: anchor(shape.num_elements(), None, Some(1), None),
                        embedded_dim: shape[2],
                        bt_len: anchor(shape[0] * shape[1], None, Some(1), None),
                        max_line_size,
                    }
                };

                let input_gen = |_key: &LearningRateGateBackwardAutotuneKey,
                                 learning_rate_base: &CubeTensor<R>,
                                 learning_rate_input: &CubeTensor<R>,
                                 output_grad: &CubeTensor<R>| {
                    (
                        learning_rate_base.copy(),
                        learning_rate_input.copy(),
                        output_grad.copy(),
                    )
                };

                static TUNER: LocalTuner<LearningRateGateBackwardAutotuneKey, CubeTuneId> =
                    local_tuner!("learning-rate-gate-backward");

                let tunables = TUNER.init(move || {
                    let launch_group =
                        TuneGroup::<LearningRateGateBackwardAutotuneKey>::new(
                            "line_size_reduce_tile",
                            |_| 1,
                        );
                    let mut set = TunableSet::new(key, input_gen);

                    for line_size in LINE_SIZE_CANDIDATES {
                        for block_size in REDUCE_BLOCK_SIZE_CANDIDATES {
                            for bt_tile in BT_TILE_CANDIDATES {
                                set = set.with(
                                    Tunable::new(
                                        format!("line_{line_size}_block_{block_size}_bt_{bt_tile}"),
                                        (move |learning_rate_base: CubeTensor<R>,
                                               learning_rate_input: CubeTensor<R>,
                                               output_grad: CubeTensor<R>| {
                                            let shape = learning_rate_input.meta.shape().clone();
                                            let embedded_dim = shape[2];
                                            let client = learning_rate_input.client.clone();
                                            let device = learning_rate_input.device.clone();
                                            let learning_rate_input_grad = empty_device::<R, F>(
                                                client.clone(),
                                                device.clone(),
                                                shape.clone(),
                                            );
                                            let bt_len = shape[0] * shape[1];
                                            let learning_rate_base_grad_shape =
                                                Shape::new([embedded_dim]);

                                            let learning_rate_base_grad = if bt_len == 0 {
                                                zeros_client::<R>(
                                                    client.clone(),
                                                    device,
                                                    learning_rate_base_grad_shape,
                                                    learning_rate_input.dtype,
                                                )
                                            } else {
                                                let num_bt_tiles = bt_len.div_ceil(bt_tile);
                                                let partial_shape =
                                                    Shape::new([num_bt_tiles, embedded_dim]);
                                                let partial_sums = empty_device::<R, F>(
                                                    client.clone(),
                                                    device.clone(),
                                                    partial_shape,
                                                );
                                                let learning_rate_base_grad = empty_device::<R, F>(
                                                    client.clone(),
                                                    device.clone(),
                                                    learning_rate_base_grad_shape,
                                                );
                                                let channel_vecs = embedded_dim / line_size;
                                                let cubes_x =
                                                    channel_vecs.div_ceil(block_size as usize)
                                                        as u32;
                                                let partial_address_type = max_address_type(&[
                                                    &learning_rate_base,
                                                    &learning_rate_input,
                                                    &output_grad,
                                                    &learning_rate_input_grad,
                                                    &partial_sums,
                                                ]);

                                                // The CUDA reference recomputes sigmoid in backward instead of storing
                                                // the forward output. This partial launch follows that shape: each worker
                                                // owns one embedded-dimension vector and a `[batch_size, context_len]`
                                                // tile, writes `learning_rate_input_grad` for every visited element, and
                                                // accumulates one partial `learning_rate_base` gradient. Partial sums use
                                                // the backend float type; if the backend is bf16 this differs from the
                                                // CUDA fixture's explicit f32 partial buffer and should be revisited with
                                                // benchmark/error evidence.
                                                // SAFETY: The public contract checks dtype/device/shape/contiguity before
                                                // primitive dispatch. Autotune only enables vector sizes that divide
                                                // `embedded_dim`, so every linear vector view covers a whole number of
                                                // embedded-dimension lanes.
                                                unsafe {
                                                    learning_rate_gate_backward_partial_kernel::launch_unchecked::<F, R>(
                                                        &client,
                                                        CubeCount::Static(cubes_x, num_bt_tiles as u32, 1),
                                                        CubeDim::new_1d(block_size),
                                                        partial_address_type,
                                                        line_size,
                                                        learning_rate_base.into_linear_view(),
                                                        learning_rate_input.into_linear_view(),
                                                        output_grad.into_linear_view(),
                                                        learning_rate_input_grad.clone().into_linear_view(),
                                                        partial_sums.clone().into_linear_view(),
                                                        channel_vecs,
                                                        bt_len,
                                                        bt_tile,
                                                    );
                                                }

                                                let finalize_address_type = max_address_type(&[
                                                    &partial_sums,
                                                    &learning_rate_base_grad,
                                                ]);
                                                // SAFETY: `partial_sums` and `learning_rate_base_grad` are allocated with
                                                // the launch shapes above, and the finalize kernel guards tuned grid
                                                // width against `channel_vecs`.
                                                unsafe {
                                                    learning_rate_gate_backward_finalize_kernel::launch_unchecked::<F, R>(
                                                        &client,
                                                        CubeCount::Static(cubes_x, 1, 1),
                                                        CubeDim::new_1d(block_size),
                                                        finalize_address_type,
                                                        line_size,
                                                        partial_sums.into_linear_view(),
                                                        learning_rate_base_grad.clone().into_linear_view(),
                                                        channel_vecs,
                                                        num_bt_tiles,
                                                    );
                                                }

                                                learning_rate_base_grad
                                            };

                                            LearningRateGateBackwardPrimitiveOutputs::<
                                                CubeBackend<R, F, I, BT>,
                                            > {
                                                learning_rate_base_grad,
                                                learning_rate_input_grad,
                                            }
                                        })
                                        .ok(),
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

                let grads_out = TUNER.execute(
                    &CubeTuneId::new(&learning_rate_input.client, &learning_rate_input.device),
                    &client,
                    tunables,
                    (learning_rate_base, learning_rate_input, output_grad),
                );

                if let Some(node) = node_learning_rate_base {
                    grads.register::<CubeBackend<R, F, I, BT>>(
                        node.id,
                        grads_out.learning_rate_base_grad,
                    );
                }
                if let Some(node) = node_learning_rate_input {
                    grads.register::<CubeBackend<R, F, I, BT>>(
                        node.id,
                        grads_out.learning_rate_input_grad,
                    );
                }
            }
        }

        let LearningRateGateForwardPrimitiveInputs {
            learning_rate_base,
            learning_rate_input,
        } = inputs;
        let output = CubeBackend::<R, F, I, BT>::fused_learning_rate_gate(
            LearningRateGateForwardPrimitiveInputs {
                learning_rate_base: learning_rate_base.primitive.clone(),
                learning_rate_input: learning_rate_input.primitive.clone(),
            },
        );

        match LearningRateGateBackward
            .prepare::<C>([
                learning_rate_base.node.clone(),
                learning_rate_input.node.clone(),
            ])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let learning_rate_input_state = prep.checkpoint(&learning_rate_input);
                prep.finish(
                    (learning_rate_input_state, learning_rate_base.primitive),
                    output,
                )
            }
            OpsKind::UnTracked(prep) => prep.finish(output),
        }
    }
}

const LINE_SIZE_CANDIDATES: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];
const REDUCE_BLOCK_SIZE_CANDIDATES: [u32; 2] = [128, 256];
const BT_TILE_CANDIDATES: [usize; 4] = [16, 32, 64, 128];

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
struct LearningRateGateBackwardAutotuneKey {
    dtype: burn::tensor::DType,
    num_elements: usize,
    embedded_dim: usize,
    bt_len: usize,
    max_line_size: usize,
}

impl core::fmt::Display for LearningRateGateBackwardAutotuneKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{:?}:{}:{}:{}:{}",
            self.dtype, self.num_elements, self.embedded_dim, self.bt_len, self.max_line_size
        )
    }
}

impl AutotuneKey for LearningRateGateBackwardAutotuneKey {}

impl<R, F, I, BT> AutotuneOutput
    for LearningRateGateBackwardPrimitiveOutputs<CubeBackend<R, F, I, BT>>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    #[cfg(feature = "autotune-checks")]
    fn check_equivalence(&self, other: Self) {
        AutotuneOutput::check_equivalence(
            &self.learning_rate_base_grad,
            other.learning_rate_base_grad,
        );
        AutotuneOutput::check_equivalence(
            &self.learning_rate_input_grad,
            other.learning_rate_input_grad,
        );
    }
}

fn max_address_type<R: CubeRuntime>(tensors: &[&CubeTensor<R>]) -> AddressType {
    tensors
        .iter()
        .map(|tensor| tensor.required_address_type())
        .max()
        .unwrap_or_default()
}

fn max_line_size_backward<R: CubeRuntime>(
    learning_rate_base: &CubeTensor<R>,
    learning_rate_input: &CubeTensor<R>,
    output_grad: &CubeTensor<R>,
) -> usize {
    let base_line_size = tensor_vector_size_parallel(
        learning_rate_base
            .client
            .io_optimized_vector_sizes(learning_rate_base.dtype.size()),
        learning_rate_base.meta.shape(),
        learning_rate_base.meta.strides(),
        0,
    );
    let input_line_size = tensor_vector_size_parallel(
        learning_rate_input
            .client
            .io_optimized_vector_sizes(learning_rate_input.dtype.size()),
        learning_rate_input.meta.shape(),
        learning_rate_input.meta.strides(),
        learning_rate_input.meta.shape().num_dims() - 1,
    );
    let output_grad_line_size = tensor_vector_size_parallel(
        output_grad
            .client
            .io_optimized_vector_sizes(output_grad.dtype.size()),
        output_grad.meta.shape(),
        output_grad.meta.strides(),
        output_grad.meta.shape().num_dims() - 1,
    );

    base_line_size
        .min(input_line_size)
        .min(output_grad_line_size)
        .max(1)
}
