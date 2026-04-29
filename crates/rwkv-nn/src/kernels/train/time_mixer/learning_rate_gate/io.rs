use burn::tensor::{Tensor, ops::FloatTensor};

use crate::kernels::{
    check::{
        KernelInputsError,
        check_axis_non_empty,
        check_same_device,
        check_same_dtype,
        check_shape,
        get_tensor_info,
    },
    train::time_mixer::learning_rate_gate::LearningRateGateBackend as Backend,
};

/// Public tensor inputs for the RWKV7 learning-rate gate.
#[derive(Debug, Clone)]
pub struct LearningRateGateForwardInputs<B: burn::tensor::backend::Backend> {
    /// Per-embedded-dimension learning-rate base shaped `[embedded_dim]`.
    pub learning_rate_base: Tensor<B, 1>,
    /// Learning-rate input shaped `[batch_size, context_len, embedded_dim]`.
    pub learning_rate_input: Tensor<B, 3>,
}

impl<B> LearningRateGateForwardInputs<B>
where
    B: Backend,
{
    pub(crate) fn to_primitive(&self) -> LearningRateGateForwardPrimitiveInputs<B> {
        LearningRateGateForwardPrimitiveInputs {
            learning_rate_base: self.learning_rate_base.clone().into_primitive().tensor(),
            learning_rate_input: self.learning_rate_input.clone().into_primitive().tensor(),
        }
    }
}

impl<B> LearningRateGateForwardInputs<B>
where
    B: Backend,
{
    pub(crate) fn check(&self) -> Result<(), KernelInputsError<B>> {
        let learning_rate_base = get_tensor_info("learning_rate_base", &self.learning_rate_base);
        let learning_rate_input = get_tensor_info("learning_rate_input", &self.learning_rate_input);
        let embedded_dim = learning_rate_input.dim(2);

        check_axis_non_empty(learning_rate_input.axis(2))?;
        check_shape(&learning_rate_base, [embedded_dim])?;
        check_same_dtype(&[&learning_rate_base, &learning_rate_input])?;
        check_same_device(&[&learning_rate_base, &learning_rate_input])?;

        Ok(())
    }
}

/// Primitive tensor inputs passed to the fused learning-rate gate kernel.
#[derive(Debug, Clone)]
pub struct LearningRateGateForwardPrimitiveInputs<B: Backend> {
    /// Primitive learning-rate base shaped `[embedded_dim]`.
    pub learning_rate_base: FloatTensor<B>,
    /// Primitive learning-rate input shaped `[batch_size, context_len, embedded_dim]`.
    pub learning_rate_input: FloatTensor<B>,
}

/// Primitive gradients produced by the fused learning-rate gate backward pass.
#[derive(Debug, Clone)]
pub struct LearningRateGateBackwardPrimitiveOutputs<B: Backend> {
    /// Gradient for `learning_rate_base`, shaped `[embedded_dim]`.
    pub learning_rate_base_grad: FloatTensor<B>,
    /// Gradient for `learning_rate_input`, shaped `[batch_size, context_len, embedded_dim]`.
    pub learning_rate_input_grad: FloatTensor<B>,
}
