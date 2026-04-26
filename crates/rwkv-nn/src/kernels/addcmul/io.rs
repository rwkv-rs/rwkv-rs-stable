use burn::tensor::{Tensor, ops::FloatTensor};

use crate::kernels::{
    addcmul::AddcmulBackend as Backend,
    check::{
        KernelInputsError,
        check_axis_non_empty,
        check_same_device,
        check_same_dtype,
        check_same_shape,
        check_shape,
        get_tensor_info,
    },
};

/// Public tensor inputs for the single-scale addcmul operation.
#[derive(Debug, Clone)]
pub struct AddcmulForwardInputs<B: Backend> {
    /// Base tensor shaped `[batch_size, context_len, embedded_dim]`.
    pub base: Tensor<B, 3>,
    /// Difference tensor shaped `[batch_size, context_len, embedded_dim]`.
    pub diff: Tensor<B, 3>,
    /// Per-embedded-dimension scale tensor shaped `[1, 1, embedded_dim]`.
    pub scale: Tensor<B, 3>,
}

impl<B> AddcmulForwardInputs<B>
where
    B: Backend,
{
    pub(crate) fn to_primitive(&self) -> AddcmulForwardPrimitiveInputs<B> {
        AddcmulForwardPrimitiveInputs {
            base: self.base.clone().into_primitive().tensor(),
            diff: self.diff.clone().into_primitive().tensor(),
            scale: self.scale.clone().into_primitive().tensor(),
        }
    }
}

impl<B> AddcmulForwardInputs<B>
where
    B: Backend,
{
    pub(crate) fn check(&self) -> Result<(), KernelInputsError<B>> {
        let base = get_tensor_info("base", &self.base);
        let diff = get_tensor_info("diff", &self.diff);
        let scale = get_tensor_info("scale", &self.scale);

        check_axis_non_empty(base.axis(2))?;
        check_same_shape(&[&base, &diff])?;
        check_shape(&scale, [1, 1, base.dim(2)])?;
        check_same_dtype(&[&base, &diff, &scale])?;
        check_same_device(&[&base, &diff, &scale])?;

        Ok(())
    }
}

/// Primitive tensor inputs passed to the backend fused addcmul kernel.
#[derive(Debug, Clone)]
pub struct AddcmulForwardPrimitiveInputs<B: Backend> {
    /// Primitive base tensor shaped `[batch_size, context_len, embedded_dim]`.
    pub base: FloatTensor<B>,
    /// Primitive difference tensor shaped `[batch_size, context_len, embedded_dim]`.
    pub diff: FloatTensor<B>,
    /// Primitive per-embedded-dimension scale tensor shaped `[1, 1, embedded_dim]`.
    pub scale: FloatTensor<B>,
}

/// Public tensor inputs for the five-scale addcmul operation.
#[derive(Debug, Clone)]
pub struct Addcmul5ForwardInputs<B: Backend> {
    /// Base tensor shaped `[batch_size, context_len, embedded_dim]`.
    pub base: Tensor<B, 3>,
    /// Difference tensor shaped `[batch_size, context_len, embedded_dim]`.
    pub diff: Tensor<B, 3>,
    /// Receptance branch scale tensor shaped `[1, 1, embedded_dim]`.
    pub receptance_scale: Tensor<B, 3>,
    /// Weight-decay branch scale tensor shaped `[1, 1, embedded_dim]`.
    pub weight_decay_scale: Tensor<B, 3>,
    /// Key branch scale tensor shaped `[1, 1, embedded_dim]`.
    pub key_scale: Tensor<B, 3>,
    /// Value branch scale tensor shaped `[1, 1, embedded_dim]`.
    pub value_scale: Tensor<B, 3>,
    /// Learning-rate branch scale tensor shaped `[1, 1, embedded_dim]`.
    pub learning_rate_scale: Tensor<B, 3>,
}

impl<B> Addcmul5ForwardInputs<B>
where
    B: Backend,
{
    pub(crate) fn to_primitive(&self) -> Addcmul5ForwardPrimitiveInputs<B> {
        Addcmul5ForwardPrimitiveInputs {
            base: self.base.clone().into_primitive().tensor(),
            diff: self.diff.clone().into_primitive().tensor(),
            receptance_scale: self.receptance_scale.clone().into_primitive().tensor(),
            weight_decay_scale: self.weight_decay_scale.clone().into_primitive().tensor(),
            key_scale: self.key_scale.clone().into_primitive().tensor(),
            value_scale: self.value_scale.clone().into_primitive().tensor(),
            learning_rate_scale: self.learning_rate_scale.clone().into_primitive().tensor(),
        }
    }
}

impl<B> Addcmul5ForwardInputs<B>
where
    B: Backend,
{
    pub(crate) fn check(&self) -> Result<(), KernelInputsError<B>> {
        let base = get_tensor_info("base", &self.base);
        let diff = get_tensor_info("diff", &self.diff);
        let receptance_scale = get_tensor_info("receptance_scale", &self.receptance_scale);
        let weight_decay_scale = get_tensor_info("weight_decay_scale", &self.weight_decay_scale);
        let key_scale = get_tensor_info("key_scale", &self.key_scale);
        let value_scale = get_tensor_info("value_scale", &self.value_scale);
        let learning_rate_scale = get_tensor_info("learning_rate_scale", &self.learning_rate_scale);
        let embedded_dim = base.dim(2);

        check_axis_non_empty(base.axis(2))?;
        check_same_shape(&[&base, &diff])?;
        check_shape(&receptance_scale, [1, 1, embedded_dim])?;
        check_shape(&weight_decay_scale, [1, 1, embedded_dim])?;
        check_shape(&key_scale, [1, 1, embedded_dim])?;
        check_shape(&value_scale, [1, 1, embedded_dim])?;
        check_shape(&learning_rate_scale, [1, 1, embedded_dim])?;
        check_same_dtype(&[
            &base,
            &diff,
            &receptance_scale,
            &weight_decay_scale,
            &key_scale,
            &value_scale,
            &learning_rate_scale,
        ])?;
        check_same_device(&[
            &base,
            &diff,
            &receptance_scale,
            &weight_decay_scale,
            &key_scale,
            &value_scale,
            &learning_rate_scale,
        ])?;

        Ok(())
    }
}

/// Primitive tensor inputs passed to the backend fused five-scale addcmul kernel.
#[derive(Debug, Clone)]
pub struct Addcmul5ForwardPrimitiveInputs<B: Backend> {
    /// Primitive base tensor shaped `[batch_size, context_len, embedded_dim]`.
    pub base: FloatTensor<B>,
    /// Primitive difference tensor shaped `[batch_size, context_len, embedded_dim]`.
    pub diff: FloatTensor<B>,
    /// Primitive receptance branch scale tensor shaped `[1, 1, embedded_dim]`.
    pub receptance_scale: FloatTensor<B>,
    /// Primitive weight-decay branch scale tensor shaped `[1, 1, embedded_dim]`.
    pub weight_decay_scale: FloatTensor<B>,
    /// Primitive key branch scale tensor shaped `[1, 1, embedded_dim]`.
    pub key_scale: FloatTensor<B>,
    /// Primitive value branch scale tensor shaped `[1, 1, embedded_dim]`.
    pub value_scale: FloatTensor<B>,
    /// Primitive learning-rate branch scale tensor shaped `[1, 1, embedded_dim]`.
    pub learning_rate_scale: FloatTensor<B>,
}

/// Outputs produced by the fused five-scale addcmul kernel.
#[derive(Clone, Debug)]
pub struct Addcmul5ForwardOutput<B: Backend> {
    /// Output for the receptance branch shaped `[batch_size, context_len, embedded_dim]`.
    pub receptance_input: Tensor<B, 3>,
    /// Output for the weight-decay branch shaped `[batch_size, context_len, embedded_dim]`.
    pub weight_decay_input: Tensor<B, 3>,
    /// Output for the key branch shaped `[batch_size, context_len, embedded_dim]`.
    pub key_input: Tensor<B, 3>,
    /// Output for the value branch shaped `[batch_size, context_len, embedded_dim]`.
    pub value_input: Tensor<B, 3>,
    /// Output for the learning-rate branch shaped `[batch_size, context_len, embedded_dim]`.
    pub learning_rate_input: Tensor<B, 3>,
}

/// Primitive outputs produced by the fused five-scale addcmul kernel.
#[derive(Clone, Debug)]
pub struct Addcmul5ForwardPrimitiveOutput<B: Backend> {
    /// Output for the receptance branch shaped `[batch_size, context_len, embedded_dim]`.
    pub receptance_input: FloatTensor<B>,
    /// Output for the weight-decay branch shaped `[batch_size, context_len, embedded_dim]`.
    pub weight_decay_input: FloatTensor<B>,
    /// Output for the key branch shaped `[batch_size, context_len, embedded_dim]`.
    pub key_input: FloatTensor<B>,
    /// Output for the value branch shaped `[batch_size, context_len, embedded_dim]`.
    pub value_input: FloatTensor<B>,
    /// Output for the learning-rate branch shaped `[batch_size, context_len, embedded_dim]`.
    pub learning_rate_input: FloatTensor<B>,
}

/// Primitive tensors required by the fused backward pass for addcmul.
#[derive(Debug, Clone)]
pub struct AddcmulBackwardPrimitiveInputs<B: Backend> {
    /// Forward difference tensor shaped `[batch_size, context_len, embedded_dim]`.
    pub diff: FloatTensor<B>,
    /// Forward per-embedded-dimension scale tensor shaped `[1, 1, embedded_dim]`.
    pub scale: FloatTensor<B>,
    /// Gradient of the addcmul output, shaped `[batch_size, context_len, embedded_dim]`.
    pub output_grad: FloatTensor<B>,
}

/// Primitive gradients produced by the fused backward pass for addcmul.
#[derive(Debug, Clone)]
pub struct AddcmulBackwardPrimitiveOutputs<B: Backend> {
    /// Gradient for `base`, shaped `[batch_size, context_len, embedded_dim]`.
    pub base_grad: FloatTensor<B>,
    /// Gradient for `diff`, shaped `[batch_size, context_len, embedded_dim]`.
    pub diff_grad: FloatTensor<B>,
    /// Gradient for `scale`, shaped `[1, 1, embedded_dim]`.
    pub scale_grad: FloatTensor<B>,
}
