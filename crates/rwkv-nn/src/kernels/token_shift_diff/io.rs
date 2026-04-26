use burn::{
    prelude::Int,
    tensor::{
        Tensor,
        ops::{FloatTensor, IntTensor},
    },
};

use crate::kernels::{
    check::{
        KernelInputsError,
        check_axis_non_empty,
        check_same_device,
        check_same_dtype,
        check_shape,
        get_tensor_info,
    },
    token_shift_diff::TokenShiftDiffBackend as Backend,
};

/// Public tensor inputs for token-shift difference.
#[derive(Debug, Clone)]
pub struct TokenShiftDiffForwardInputs<B: Backend> {
    /// Embedded context shaped `[batch_size, context_len, embedded_dim]`.
    pub embedded_context: Tensor<B, 3>,
    /// Full token-shift state shaped `[full_batch_size, embedded_dim]`.
    pub embedded_token_shift: Tensor<B, 2>,
    /// Active batch row ids shaped `[batch_size]`.
    pub batch_ids: Tensor<B, 1, Int>,
}

impl<B> TokenShiftDiffForwardInputs<B>
where
    B: Backend,
{
    pub(crate) fn to_primitive(&self) -> TokenShiftDiffForwardPrimitiveInputs<B> {
        TokenShiftDiffForwardPrimitiveInputs {
            embedded_context: self.embedded_context.clone().into_primitive().tensor(),
            embedded_token_shift: self.embedded_token_shift.clone().into_primitive().tensor(),
            batch_ids: self.batch_ids.clone().into_primitive(),
        }
    }
}

impl<B> TokenShiftDiffForwardInputs<B>
where
    B: Backend,
{
    pub(crate) fn check(&self) -> Result<(), KernelInputsError<B>> {
        let embedded_context = get_tensor_info("embedded_context", &self.embedded_context);
        let embedded_token_shift =
            get_tensor_info("embedded_token_shift", &self.embedded_token_shift);
        let batch_ids = get_tensor_info("batch_ids", &self.batch_ids);

        check_axis_non_empty(embedded_context.axis(2))?;
        check_shape(
            &embedded_token_shift,
            [embedded_token_shift.dim(0), embedded_context.dim(2)],
        )?;
        check_shape(&batch_ids, [embedded_context.dim(0)])?;
        check_same_dtype(&[&embedded_context, &embedded_token_shift])?;
        check_same_device(&[&embedded_context, &embedded_token_shift, &batch_ids])?;

        Ok(())
    }
}

/// Primitive tensor inputs passed to the fused token-shift difference kernel.
#[derive(Debug, Clone)]
pub struct TokenShiftDiffForwardPrimitiveInputs<B: Backend> {
    /// Primitive embedded context shaped `[batch_size, context_len, embedded_dim]`.
    pub embedded_context: FloatTensor<B>,
    /// Primitive full token-shift state shaped `[full_batch_size, embedded_dim]`.
    pub embedded_token_shift: FloatTensor<B>,
    /// Primitive active batch row ids shaped `[batch_size]`.
    pub batch_ids: IntTensor<B>,
}

/// Outputs produced by token-shift difference.
#[derive(Clone, Debug)]
pub struct TokenShiftDiffForwardOutput<B: Backend> {
    /// Difference tensor shaped `[batch_size, context_len, embedded_dim]`.
    pub token_shifted_diff: Tensor<B, 3>,
    /// Updated token-shift state shaped `[full_batch_size, embedded_dim]`.
    pub next_token_shift: Tensor<B, 2>,
}

/// Primitive outputs produced by the fused token-shift difference kernel.
#[derive(Clone, Debug)]
pub struct TokenShiftDiffForwardPrimitiveOutput<B: Backend> {
    /// Primitive difference tensor shaped `[batch_size, context_len, embedded_dim]`.
    pub token_shifted_diff: FloatTensor<B>,
    /// Primitive updated token-shift state shaped `[full_batch_size, embedded_dim]`.
    pub next_token_shift: FloatTensor<B>,
}
