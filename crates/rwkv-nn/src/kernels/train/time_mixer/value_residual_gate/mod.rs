mod backward;
mod forward;
/// Input containers for the value residual gate kernel.
pub mod io;
mod kernel;

use burn::tensor::{Tensor, TensorPrimitive, activation::sigmoid, ops::FloatTensor};
use burn_cubecl::{
    CubeBackend,
    CubeElement,
    CubeRuntime,
    FloatElement,
    IntElement,
    element::BoolElement,
};

use crate::kernels::train::{
    layout::assert_linear_readable,
    time_mixer::value_residual_gate::io::{
        ValueResidualGateForwardInputs,
        ValueResidualGateForwardPrimitiveInputs,
    },
};

/// Backend primitive capability for the fused value residual gate.
pub trait ValueResidualGateBackend: burn::tensor::backend::Backend {
    /// Runs the RWKV7 value residual gate as one fused primitive operation.
    fn fused_value_residual_gate(
        inputs: ValueResidualGateForwardPrimitiveInputs<Self>,
    ) -> FloatTensor<Self>;
}

/// Autodiff backend marker for the value residual gate.
pub trait AutodiffBackend:
    ValueResidualGateBackend + burn::tensor::backend::AutodiffBackend
{
}

impl<B> AutodiffBackend for B where
    B: ValueResidualGateBackend + burn::tensor::backend::AutodiffBackend
{
}

impl<R, F, I, BT> ValueResidualGateBackend for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
{
    fn fused_value_residual_gate(
        inputs: ValueResidualGateForwardPrimitiveInputs<Self>,
    ) -> FloatTensor<Self> {
        assert_linear_readable("value", &inputs.value);
        assert_linear_readable("value_from_first_cell", &inputs.value_from_first_cell);
        assert_linear_readable("gate_base", &inputs.gate_base);
        assert_linear_readable("gate_input", &inputs.gate_input);

        forward::fused_value_residual_gate::<R, F, I, BT>(inputs)
    }
}

/// Runs the fused RWKV7 value residual gate after validating the public input contract.
///
/// `value`, `value_from_first_cell`, and `gate_input` must be contiguous and shaped
/// `[batch_size, context_len, embedded_dim]`. `gate_base` must be contiguous and shaped
/// `[embedded_dim]`.
///
/// Mathematically:
/// `gate[b, t, e] = sigmoid(gate_base[e] + gate_input[b, t, e])`;
/// `output[b, t, e] = value[b, t, e] + (value_from_first_cell[b, t, e] - value[b, t, e]) *
/// gate[b, t, e]`.
///
/// This ports the RWKV-LM `tmix_vres_gate` fast path using repository terminology. The custom
/// path fuses the broadcast add, sigmoid, and residual blend into one elementwise kernel and keeps
/// the gate-base broadcast as index math. Backward uses CubeCL kernels for the elementwise
/// gradients and the gate-base reduction across `[batch_size, context_len]`.
pub fn value_residual_gate_custom<B: ValueResidualGateBackend>(
    inputs: ValueResidualGateForwardInputs<B>,
) -> Tensor<B, 3> {
    inputs.check().unwrap();
    let output = B::fused_value_residual_gate(inputs.to_primitive());

    Tensor::from_primitive(TensorPrimitive::Float(output))
}

/// Computes the value residual gate with regular Burn tensor operations.
///
/// This is the semantic reference for correctness and gradient tests. Burn fusion may simplify the
/// generic expression graph, while the custom path exposes an explicit CubeCL primitive with
/// autotuned vector width.
pub fn value_residual_gate_reference<B: ValueResidualGateBackend>(
    inputs: ValueResidualGateForwardInputs<B>,
) -> Tensor<B, 3> {
    let gate =
        sigmoid(inputs.gate_input + inputs.gate_base.unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0));

    inputs.value.clone() + (inputs.value_from_first_cell - inputs.value) * gate
}

/// Convenience wrapper for the fused value residual gate path.
pub fn value_residual_gate<B: ValueResidualGateBackend>(
    value: Tensor<B, 3>,
    value_from_first_cell: Tensor<B, 3>,
    gate_base: Tensor<B, 1>,
    gate_input: Tensor<B, 3>,
) -> Tensor<B, 3> {
    value_residual_gate_custom(ValueResidualGateForwardInputs {
        value,
        value_from_first_cell,
        gate_base,
        gate_input,
    })
}
