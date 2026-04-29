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
        calculate_cube_count_elemwise,
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

use crate::kernels::train::time_mixer::value_residual_gate::{
    ValueResidualGateBackend,
    io::{ValueResidualGateBackwardPrimitiveOutputs, ValueResidualGateForwardPrimitiveInputs},
    kernel::{
        value_residual_gate_backward_elementwise_kernel,
        value_residual_gate_backward_elementwise_pow2_kernel,
        value_residual_gate_base_reduce_finalize_kernel,
        value_residual_gate_base_reduce_partial_kernel,
    },
};

impl<R, F, I, BT, C> ValueResidualGateBackend for Autodiff<CubeBackend<R, F, I, BT>, C>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
    C: CheckpointStrategy,
{
    fn fused_value_residual_gate(
        inputs: ValueResidualGateForwardPrimitiveInputs<Self>,
    ) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct ValueResidualGateBackward;

        impl<R, F, I, BT> Backward<CubeBackend<R, F, I, BT>, 4> for ValueResidualGateBackward
        where
            R: CubeRuntime,
            F: FloatElement + CubeElement,
            I: IntElement,
            BT: BoolElement,
        {
            type State = (
                NodeId,
                NodeId,
                NodeId,
                FloatTensor<CubeBackend<R, F, I, BT>>,
            );

            fn backward(
                self,
                ops: Ops<Self::State, 4>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [
                    node_value,
                    node_value_from_first_cell,
                    node_gate_base,
                    node_gate_input,
                ] = ops.parents;
                let output_grad = grads.consume::<CubeBackend<R, F, I, BT>>(&ops.node);
                let (value_state, value_from_first_cell_state, gate_input_state, gate_base) =
                    ops.state;
                let value: FloatTensor<CubeBackend<R, F, I, BT>> =
                    checkpointer.retrieve_node_output(value_state);
                let value_from_first_cell: FloatTensor<CubeBackend<R, F, I, BT>> =
                    checkpointer.retrieve_node_output(value_from_first_cell_state);
                let gate_input: FloatTensor<CubeBackend<R, F, I, BT>> =
                    checkpointer.retrieve_node_output(gate_input_state);

                assert!(value.is_contiguous(), "value must be contiguous");
                assert!(
                    value_from_first_cell.is_contiguous(),
                    "value_from_first_cell must be contiguous"
                );
                assert!(gate_base.is_contiguous(), "gate_base must be contiguous");
                assert!(gate_input.is_contiguous(), "gate_input must be contiguous");
                assert!(
                    output_grad.is_contiguous(),
                    "output_grad must be contiguous"
                );

                let client = value.client.clone();
                let key = |value: &CubeTensor<R>,
                           value_from_first_cell: &CubeTensor<R>,
                           gate_base: &CubeTensor<R>,
                           gate_input: &CubeTensor<R>,
                           output_grad: &CubeTensor<R>| {
                    let shape = value.meta.shape();

                    ValueResidualGateBackwardAutotuneKey {
                        dtype: value.dtype,
                        num_elements: anchor(shape.num_elements(), None, Some(1), None),
                        embedded_dim: shape[2],
                        bt_len: anchor(shape[0] * shape[1], None, Some(1), None),
                        max_line_size: max_line_size_backward(
                            value,
                            value_from_first_cell,
                            gate_base,
                            gate_input,
                            output_grad,
                        ),
                    }
                };

                let input_gen = |_key: &ValueResidualGateBackwardAutotuneKey,
                                 value: &CubeTensor<R>,
                                 value_from_first_cell: &CubeTensor<R>,
                                 gate_base: &CubeTensor<R>,
                                 gate_input: &CubeTensor<R>,
                                 output_grad: &CubeTensor<R>| {
                    (
                        value.copy(),
                        value_from_first_cell.copy(),
                        gate_base.copy(),
                        gate_input.copy(),
                        output_grad.copy(),
                    )
                };

                static TUNER: LocalTuner<ValueResidualGateBackwardAutotuneKey, CubeTuneId> =
                    local_tuner!("value-residual-gate-backward");

                let tunables = TUNER.init(move || {
                    let launch_group =
                        TuneGroup::<ValueResidualGateBackwardAutotuneKey>::new(
                            "line_size_reduce_tile_pow2",
                            |_| 1,
                        );
                    let mut set = TunableSet::new(key, input_gen);

                    for line_size in LINE_SIZE_CANDIDATES {
                        for block_size in REDUCE_BLOCK_SIZE_CANDIDATES {
                            for bt_tile in BT_TILE_CANDIDATES {
                                for use_pow2_index in [false, true] {
                                    set = set.with(
                                        Tunable::new(
                                            format!(
                                                "line_{line_size}_block_{block_size}_bt_{bt_tile}_pow2_{use_pow2_index}"
                                            ),
                                            (move |value: CubeTensor<R>,
                                                   value_from_first_cell: CubeTensor<R>,
                                                   gate_base: CubeTensor<R>,
                                                   gate_input: CubeTensor<R>,
                                                   output_grad: CubeTensor<R>| {
                                                let shape = value.meta.shape().clone();
                                                let embedded_dim = shape[2];
                                                let client = value.client.clone();
                                                let device = value.device.clone();
                                                let dtype = value.dtype;
                                                let value_grad = empty_device::<R, F>(
                                                    client.clone(),
                                                    device.clone(),
                                                    shape.clone(),
                                                );
                                                let value_from_first_cell_grad =
                                                    empty_device::<R, F>(
                                                        client.clone(),
                                                        device.clone(),
                                                        shape.clone(),
                                                    );
                                                let gate_input_grad = empty_device::<R, F>(
                                                    client.clone(),
                                                    device.clone(),
                                                    shape.clone(),
                                                );

                                                if shape.num_elements() > 0 {
                                                    let working_units =
                                                        shape.num_elements() / line_size;
                                                    let cube_dim =
                                                        CubeDim::new(&client, working_units);
                                                    let cube_count =
                                                        calculate_cube_count_elemwise(
                                                            &client,
                                                            working_units,
                                                            cube_dim,
                                                        );
                                                    let elementwise_address_type =
                                                        max_address_type(&[
                                                            &value,
                                                            &value_from_first_cell,
                                                            &gate_base,
                                                            &gate_input,
                                                            &output_grad,
                                                            &value_grad,
                                                            &value_from_first_cell_grad,
                                                            &gate_input_grad,
                                                        ]);

                                                    // One work unit computes a vector of contiguous embedded lanes. The
                                                    // CUDA algebra is used for `value_grad`: derive it from
                                                    // `output_grad - value_from_first_cell_grad` after computing the gated
                                                    // first-cell branch. The pre-activation gradient is written as
                                                    // `gate_input_grad`, then reduced in separate kernels so elementwise
                                                    // parallelism stays wide.
                                                    // SAFETY: The public input contract checks dtype/device/shape and the
                                                    // primitive dispatch checks contiguity. The autotune group only keeps
                                                    // vector widths that divide `embedded_dim`, so each linear view covers
                                                    // a whole number of embedded vectors.
                                                    unsafe {
                                                        if use_pow2_index {
                                                            let embedded_vec_mask =
                                                                embedded_dim / line_size - 1;
                                                            value_residual_gate_backward_elementwise_pow2_kernel::launch_unchecked::<F, R>(
                                                                &client,
                                                                cube_count,
                                                                cube_dim,
                                                                elementwise_address_type,
                                                                line_size,
                                                                value.clone().into_linear_view_like(&gate_input_grad),
                                                                value_from_first_cell.clone().into_linear_view_like(&gate_input_grad),
                                                                gate_base.clone().into_linear_view(),
                                                                gate_input.clone().into_linear_view_like(&gate_input_grad),
                                                                output_grad.clone().into_linear_view_like(&gate_input_grad),
                                                                value_grad.clone().into_linear_view(),
                                                                value_from_first_cell_grad.clone().into_linear_view(),
                                                                gate_input_grad.clone().into_linear_view(),
                                                                embedded_vec_mask,
                                                            );
                                                        } else {
                                                            value_residual_gate_backward_elementwise_kernel::launch_unchecked::<F, R>(
                                                                &client,
                                                                cube_count,
                                                                cube_dim,
                                                                elementwise_address_type,
                                                                line_size,
                                                                value.clone().into_linear_view_like(&gate_input_grad),
                                                                value_from_first_cell.clone().into_linear_view_like(&gate_input_grad),
                                                                gate_base.clone().into_linear_view(),
                                                                gate_input.clone().into_linear_view_like(&gate_input_grad),
                                                                output_grad.clone().into_linear_view_like(&gate_input_grad),
                                                                value_grad.clone().into_linear_view(),
                                                                value_from_first_cell_grad.clone().into_linear_view(),
                                                                gate_input_grad.clone().into_linear_view(),
                                                            );
                                                        }
                                                    }
                                                }

                                                let bt_len = shape[0] * shape[1];
                                                let gate_base_grad_shape = Shape::new([embedded_dim]);
                                                let gate_base_grad = if bt_len == 0 {
                                                    zeros_client::<R>(
                                                        client.clone(),
                                                        device,
                                                        gate_base_grad_shape,
                                                        dtype,
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
                                                    let gate_base_grad = empty_device::<R, F>(
                                                        client.clone(),
                                                        device.clone(),
                                                        gate_base_grad_shape,
                                                    );
                                                    let channel_vecs = embedded_dim / line_size;
                                                    let cubes_x =
                                                        channel_vecs.div_ceil(block_size as usize)
                                                            as u32;
                                                    let partial_address_type = max_address_type(&[
                                                        &gate_input_grad,
                                                        &partial_sums,
                                                    ]);

                                                    // Reduction is isolated to `gate_base_grad[e] =
                                                    // sum_{batch_size,context_len} gate_input_grad[b,t,e]`.
                                                    // Partial workers stream contiguous vectors for one embedded vector
                                                    // across a BT tile and keep only an accumulator live. Finalize then
                                                    // folds the tile partials for each embedded vector.
                                                    // SAFETY: `gate_input_grad`, `partial_sums`, and `gate_base_grad` are
                                                    // allocated with shapes matching `channel_vecs` and `num_bt_tiles`; both
                                                    // reduction kernels guard tuned grid width against `channel_vecs`.
                                                    unsafe {
                                                        value_residual_gate_base_reduce_partial_kernel::launch_unchecked::<F, R>(
                                                            &client,
                                                            CubeCount::Static(cubes_x, num_bt_tiles as u32, 1),
                                                            CubeDim::new_1d(block_size),
                                                            partial_address_type,
                                                            line_size,
                                                            gate_input_grad.clone().into_linear_view(),
                                                            partial_sums.clone().into_linear_view(),
                                                            channel_vecs,
                                                            bt_len,
                                                            bt_tile,
                                                        );
                                                    }

                                                    let finalize_address_type = max_address_type(&[
                                                        &partial_sums,
                                                        &gate_base_grad,
                                                    ]);
                                                    unsafe {
                                                        value_residual_gate_base_reduce_finalize_kernel::launch_unchecked::<F, R>(
                                                            &client,
                                                            CubeCount::Static(cubes_x, 1, 1),
                                                            CubeDim::new_1d(block_size),
                                                            finalize_address_type,
                                                            line_size,
                                                            partial_sums.into_linear_view(),
                                                            gate_base_grad.clone().into_linear_view(),
                                                            channel_vecs,
                                                            num_bt_tiles,
                                                        );
                                                    }

                                                    gate_base_grad
                                                };

                                                ValueResidualGateBackwardPrimitiveOutputs::<
                                                    CubeBackend<R, F, I, BT>,
                                                > {
                                                    value_grad,
                                                    value_from_first_cell_grad,
                                                    gate_base_grad,
                                                    gate_input_grad,
                                                }
                                            })
                                            .ok(),
                                        )
                                        .group(&launch_group, move |key| {
                                            let channel_vecs = key.embedded_dim / line_size;
                                            let valid_line_size = line_size <= key.max_line_size
                                                && key.embedded_dim.is_multiple_of(line_size);
                                            let valid_index = !use_pow2_index
                                                || channel_vecs.is_power_of_two();

                                            if valid_line_size && valid_index {
                                                1
                                            } else {
                                                -1
                                            }
                                        }),
                                    );
                                }
                            }
                        }
                    }

                    set
                });

                let grads_out = TUNER.execute(
                    &CubeTuneId::new(&value.client, &value.device),
                    &client,
                    tunables,
                    (
                        value,
                        value_from_first_cell,
                        gate_base,
                        gate_input,
                        output_grad,
                    ),
                );

                if let Some(node) = node_value {
                    grads.register::<CubeBackend<R, F, I, BT>>(node.id, grads_out.value_grad);
                }
                if let Some(node) = node_value_from_first_cell {
                    grads.register::<CubeBackend<R, F, I, BT>>(
                        node.id,
                        grads_out.value_from_first_cell_grad,
                    );
                }
                if let Some(node) = node_gate_base {
                    grads.register::<CubeBackend<R, F, I, BT>>(node.id, grads_out.gate_base_grad);
                }
                if let Some(node) = node_gate_input {
                    grads.register::<CubeBackend<R, F, I, BT>>(node.id, grads_out.gate_input_grad);
                }
            }
        }

        let ValueResidualGateForwardPrimitiveInputs {
            value,
            value_from_first_cell,
            gate_base,
            gate_input,
        } = inputs;

        let output = CubeBackend::<R, F, I, BT>::fused_value_residual_gate(
            ValueResidualGateForwardPrimitiveInputs {
                value: value.primitive.clone(),
                value_from_first_cell: value_from_first_cell.primitive.clone(),
                gate_base: gate_base.primitive.clone(),
                gate_input: gate_input.primitive.clone(),
            },
        );

        match ValueResidualGateBackward
            .prepare::<C>([
                value.node.clone(),
                value_from_first_cell.node.clone(),
                gate_base.node.clone(),
                gate_input.node.clone(),
            ])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let value_state = prep.checkpoint(&value);
                let value_from_first_cell_state = prep.checkpoint(&value_from_first_cell);
                let gate_input_state = prep.checkpoint(&gate_input);
                prep.finish(
                    (
                        value_state,
                        value_from_first_cell_state,
                        gate_input_state,
                        gate_base.primitive,
                    ),
                    output,
                )
            }
            OpsKind::UnTracked(prep) => prep.finish(output),
        }
    }
}

const LINE_SIZE_CANDIDATES: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];
const REDUCE_BLOCK_SIZE_CANDIDATES: [u32; 3] = [64, 128, 256];
const BT_TILE_CANDIDATES: [usize; 3] = [64, 128, 256];

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
struct ValueResidualGateBackwardAutotuneKey {
    dtype: burn::tensor::DType,
    num_elements: usize,
    embedded_dim: usize,
    bt_len: usize,
    max_line_size: usize,
}

impl core::fmt::Display for ValueResidualGateBackwardAutotuneKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{:?}:{}:{}:{}:{}",
            self.dtype, self.num_elements, self.embedded_dim, self.bt_len, self.max_line_size
        )
    }
}

impl AutotuneKey for ValueResidualGateBackwardAutotuneKey {}

impl<R, F, I, BT> AutotuneOutput
    for ValueResidualGateBackwardPrimitiveOutputs<CubeBackend<R, F, I, BT>>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    #[cfg(feature = "autotune-checks")]
    fn check_equivalence(&self, other: Self) {
        AutotuneOutput::check_equivalence(&self.value_grad, other.value_grad);
        AutotuneOutput::check_equivalence(
            &self.value_from_first_cell_grad,
            other.value_from_first_cell_grad,
        );
        AutotuneOutput::check_equivalence(&self.gate_base_grad, other.gate_base_grad);
        AutotuneOutput::check_equivalence(&self.gate_input_grad, other.gate_input_grad);
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
    value: &CubeTensor<R>,
    value_from_first_cell: &CubeTensor<R>,
    gate_base: &CubeTensor<R>,
    gate_input: &CubeTensor<R>,
    output_grad: &CubeTensor<R>,
) -> usize {
    let value_line_size = tensor_vector_size_parallel(
        value.client.io_optimized_vector_sizes(value.dtype.size()),
        value.meta.shape(),
        value.meta.strides(),
        value.meta.shape().num_dims() - 1,
    );
    let value_from_first_cell_line_size = tensor_vector_size_parallel(
        value_from_first_cell
            .client
            .io_optimized_vector_sizes(value_from_first_cell.dtype.size()),
        value_from_first_cell.meta.shape(),
        value_from_first_cell.meta.strides(),
        value_from_first_cell.meta.shape().num_dims() - 1,
    );
    let gate_base_line_size = tensor_vector_size_parallel(
        gate_base
            .client
            .io_optimized_vector_sizes(gate_base.dtype.size()),
        gate_base.meta.shape(),
        gate_base.meta.strides(),
        0,
    );
    let gate_input_line_size = tensor_vector_size_parallel(
        gate_input
            .client
            .io_optimized_vector_sizes(gate_input.dtype.size()),
        gate_input.meta.shape(),
        gate_input.meta.strides(),
        gate_input.meta.shape().num_dims() - 1,
    );
    let output_grad_line_size = tensor_vector_size_parallel(
        output_grad
            .client
            .io_optimized_vector_sizes(output_grad.dtype.size()),
        output_grad.meta.shape(),
        output_grad.meta.strides(),
        output_grad.meta.shape().num_dims() - 1,
    );

    value_line_size
        .min(value_from_first_cell_line_size)
        .min(gate_base_line_size)
        .min(gate_input_line_size)
        .min(output_grad_line_size)
        .max(1)
}
