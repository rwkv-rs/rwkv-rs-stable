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
    train::time_mixer::weight_decay_transform::WeightDecayTransformBackend as Backend,
};

/// Public tensor inputs for the RWKV7 weight-decay transform.
#[derive(Debug, Clone)]
pub struct WeightDecayTransformForwardInputs<B: Backend> {
    /// Per-embedded-dimension weight-decay base shaped `[embedded_dim]`.
    pub weight_decay_base: Tensor<B, 1>,
    /// Weight-decay LoRA input shaped `[batch_size, context_len, embedded_dim]`.
    pub weight_decay_input: Tensor<B, 3>,
}

impl<B> WeightDecayTransformForwardInputs<B>
where
    B: Backend,
{
    pub(crate) fn to_primitive(&self) -> WeightDecayTransformForwardPrimitiveInputs<B> {
        WeightDecayTransformForwardPrimitiveInputs {
            weight_decay_base: self.weight_decay_base.clone().into_primitive().tensor(),
            weight_decay_input: self.weight_decay_input.clone().into_primitive().tensor(),
        }
    }

    pub(crate) fn check(&self) -> Result<(), KernelInputsError<B>> {
        let weight_decay_base = get_tensor_info("weight_decay_base", &self.weight_decay_base);
        let weight_decay_input = get_tensor_info("weight_decay_input", &self.weight_decay_input);
        let embedded_dim = weight_decay_input.dim(2);

        check_axis_non_empty(weight_decay_input.axis(2))?;
        check_shape(&weight_decay_base, [embedded_dim])?;
        check_same_dtype(&[&weight_decay_base, &weight_decay_input])?;
        check_same_device(&[&weight_decay_base, &weight_decay_input])?;

        Ok(())
    }
}

/// Primitive tensor inputs passed to the fused weight-decay transform kernel.
#[derive(Debug, Clone)]
pub struct WeightDecayTransformForwardPrimitiveInputs<B: Backend> {
    /// Primitive weight-decay base shaped `[embedded_dim]`.
    pub weight_decay_base: FloatTensor<B>,
    /// Primitive weight-decay input shaped `[batch_size, context_len, embedded_dim]`.
    pub weight_decay_input: FloatTensor<B>,
}
