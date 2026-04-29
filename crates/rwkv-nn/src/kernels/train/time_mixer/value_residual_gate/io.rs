use burn::tensor::{Tensor, ops::FloatTensor};

use crate::kernels::{
    check::{
        KernelInputsError,
        check_axis_non_empty,
        check_same_device,
        check_same_dtype,
        check_same_shape,
        check_shape,
        get_tensor_info,
    },
    train::time_mixer::value_residual_gate::ValueResidualGateBackend as Backend,
};

/// Public tensor inputs for the RWKV7 value residual gate.
#[derive(Debug, Clone)]
pub struct ValueResidualGateForwardInputs<B: Backend> {
    /// Current value tensor shaped `[batch_size, context_len, embedded_dim]`.
    pub value: Tensor<B, 3>,
    /// First-cell value tensor shaped `[batch_size, context_len, embedded_dim]`.
    pub value_from_first_cell: Tensor<B, 3>,
    /// Per-embedded-dimension gate base shaped `[embedded_dim]`.
    pub gate_base: Tensor<B, 1>,
    /// Gate input shaped `[batch_size, context_len, embedded_dim]`.
    pub gate_input: Tensor<B, 3>,
}

impl<B> ValueResidualGateForwardInputs<B>
where
    B: Backend,
{
    pub(crate) fn to_primitive(&self) -> ValueResidualGateForwardPrimitiveInputs<B> {
        ValueResidualGateForwardPrimitiveInputs {
            value: self.value.clone().into_primitive().tensor(),
            value_from_first_cell: self.value_from_first_cell.clone().into_primitive().tensor(),
            gate_base: self.gate_base.clone().into_primitive().tensor(),
            gate_input: self.gate_input.clone().into_primitive().tensor(),
        }
    }
}

impl<B> ValueResidualGateForwardInputs<B>
where
    B: Backend,
{
    pub(crate) fn check(&self) -> Result<(), KernelInputsError<B>> {
        let value = get_tensor_info("value", &self.value);
        let value_from_first_cell =
            get_tensor_info("value_from_first_cell", &self.value_from_first_cell);
        let gate_base = get_tensor_info("gate_base", &self.gate_base);
        let gate_input = get_tensor_info("gate_input", &self.gate_input);
        let embedded_dim = value.dim(2);

        check_axis_non_empty(value.axis(2))?;
        check_same_shape(&[&value, &value_from_first_cell, &gate_input])?;
        check_shape(&gate_base, [embedded_dim])?;
        check_same_dtype(&[&value, &value_from_first_cell, &gate_base, &gate_input])?;
        check_same_device(&[&value, &value_from_first_cell, &gate_base, &gate_input])?;

        Ok(())
    }
}

/// Primitive tensor inputs passed to the fused value residual gate kernel.
#[derive(Debug, Clone)]
pub struct ValueResidualGateForwardPrimitiveInputs<B: Backend> {
    /// Primitive current value tensor shaped `[batch_size, context_len, embedded_dim]`.
    pub value: FloatTensor<B>,
    /// Primitive first-cell value tensor shaped `[batch_size, context_len, embedded_dim]`.
    pub value_from_first_cell: FloatTensor<B>,
    /// Primitive per-embedded-dimension gate base shaped `[embedded_dim]`.
    pub gate_base: FloatTensor<B>,
    /// Primitive gate input shaped `[batch_size, context_len, embedded_dim]`.
    pub gate_input: FloatTensor<B>,
}

/// Primitive tensor gradients produced by the fused value residual gate backward path.
#[derive(Debug, Clone)]
pub(crate) struct ValueResidualGateBackwardPrimitiveOutputs<B: Backend> {
    /// Gradient for `value`, shaped `[batch_size, context_len, embedded_dim]`.
    pub value_grad: FloatTensor<B>,
    /// Gradient for `value_from_first_cell`, shaped `[batch_size, context_len, embedded_dim]`.
    pub value_from_first_cell_grad: FloatTensor<B>,
    /// Gradient for `gate_base`, shaped `[embedded_dim]`.
    pub gate_base_grad: FloatTensor<B>,
    /// Gradient for `gate_input`, shaped `[batch_size, context_len, embedded_dim]`.
    pub gate_input_grad: FloatTensor<B>,
}
