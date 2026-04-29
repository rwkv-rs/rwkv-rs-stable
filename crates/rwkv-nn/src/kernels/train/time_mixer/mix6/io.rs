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
    train::time_mixer::mix6::Mix6Backend as Backend,
};

/// Public tensor inputs for RWKV7 pretrain mix6.
#[derive(Debug, Clone)]
pub struct Mix6ForwardInputs<B: Backend> {
    /// Embedded context shaped `[batch_size, context_len, embedded_dim]`.
    pub embedded_context: Tensor<B, 3>,
    /// Receptance branch scale shaped `[1, 1, embedded_dim]`.
    pub receptance_scale: Tensor<B, 3>,
    /// Weight-decay branch scale shaped `[1, 1, embedded_dim]`.
    pub weight_decay_scale: Tensor<B, 3>,
    /// Key branch scale shaped `[1, 1, embedded_dim]`.
    pub key_scale: Tensor<B, 3>,
    /// Value branch scale shaped `[1, 1, embedded_dim]`.
    pub value_scale: Tensor<B, 3>,
    /// Learning-rate branch scale shaped `[1, 1, embedded_dim]`.
    pub learning_rate_scale: Tensor<B, 3>,
    /// Gate branch scale shaped `[1, 1, embedded_dim]`.
    pub gate_scale: Tensor<B, 3>,
}

impl<B> Mix6ForwardInputs<B>
where
    B: Backend,
{
    pub(crate) fn to_primitive(&self) -> Mix6ForwardPrimitiveInputs<B> {
        Mix6ForwardPrimitiveInputs {
            embedded_context: self.embedded_context.clone().into_primitive().tensor(),
            receptance_scale: self.receptance_scale.clone().into_primitive().tensor(),
            weight_decay_scale: self.weight_decay_scale.clone().into_primitive().tensor(),
            key_scale: self.key_scale.clone().into_primitive().tensor(),
            value_scale: self.value_scale.clone().into_primitive().tensor(),
            learning_rate_scale: self.learning_rate_scale.clone().into_primitive().tensor(),
            gate_scale: self.gate_scale.clone().into_primitive().tensor(),
        }
    }

    pub(crate) fn check(&self) -> Result<(), KernelInputsError<B>> {
        let embedded_context = get_tensor_info("embedded_context", &self.embedded_context);
        let receptance_scale = get_tensor_info("receptance_scale", &self.receptance_scale);
        let weight_decay_scale = get_tensor_info("weight_decay_scale", &self.weight_decay_scale);
        let key_scale = get_tensor_info("key_scale", &self.key_scale);
        let value_scale = get_tensor_info("value_scale", &self.value_scale);
        let learning_rate_scale = get_tensor_info("learning_rate_scale", &self.learning_rate_scale);
        let gate_scale = get_tensor_info("gate_scale", &self.gate_scale);
        let embedded_dim = embedded_context.dim(2);

        check_axis_non_empty(embedded_context.axis(2))?;
        check_shape(&receptance_scale, [1, 1, embedded_dim])?;
        check_shape(&weight_decay_scale, [1, 1, embedded_dim])?;
        check_shape(&key_scale, [1, 1, embedded_dim])?;
        check_shape(&value_scale, [1, 1, embedded_dim])?;
        check_shape(&learning_rate_scale, [1, 1, embedded_dim])?;
        check_shape(&gate_scale, [1, 1, embedded_dim])?;
        check_same_dtype(&[
            &embedded_context,
            &receptance_scale,
            &weight_decay_scale,
            &key_scale,
            &value_scale,
            &learning_rate_scale,
            &gate_scale,
        ])?;
        check_same_device(&[
            &embedded_context,
            &receptance_scale,
            &weight_decay_scale,
            &key_scale,
            &value_scale,
            &learning_rate_scale,
            &gate_scale,
        ])?;

        Ok(())
    }
}

/// Primitive tensor inputs passed to the fused mix6 kernel.
#[derive(Debug, Clone)]
pub struct Mix6ForwardPrimitiveInputs<B: Backend> {
    /// Primitive embedded context shaped `[batch_size, context_len, embedded_dim]`.
    pub embedded_context: FloatTensor<B>,
    /// Primitive receptance scale shaped `[1, 1, embedded_dim]`.
    pub receptance_scale: FloatTensor<B>,
    /// Primitive weight-decay scale shaped `[1, 1, embedded_dim]`.
    pub weight_decay_scale: FloatTensor<B>,
    /// Primitive key scale shaped `[1, 1, embedded_dim]`.
    pub key_scale: FloatTensor<B>,
    /// Primitive value scale shaped `[1, 1, embedded_dim]`.
    pub value_scale: FloatTensor<B>,
    /// Primitive learning-rate scale shaped `[1, 1, embedded_dim]`.
    pub learning_rate_scale: FloatTensor<B>,
    /// Primitive gate scale shaped `[1, 1, embedded_dim]`.
    pub gate_scale: FloatTensor<B>,
}

/// Public outputs produced by the fused mix6 operation.
#[derive(Clone, Debug)]
pub struct Mix6ForwardOutput<B: Backend> {
    /// Receptance branch input shaped `[batch_size, context_len, embedded_dim]`.
    pub receptance_input: Tensor<B, 3>,
    /// Weight-decay branch input shaped `[batch_size, context_len, embedded_dim]`.
    pub weight_decay_input: Tensor<B, 3>,
    /// Key branch input shaped `[batch_size, context_len, embedded_dim]`.
    pub key_input: Tensor<B, 3>,
    /// Value branch input shaped `[batch_size, context_len, embedded_dim]`.
    pub value_input: Tensor<B, 3>,
    /// Learning-rate branch input shaped `[batch_size, context_len, embedded_dim]`.
    pub learning_rate_input: Tensor<B, 3>,
    /// Gate branch input shaped `[batch_size, context_len, embedded_dim]`.
    pub gate_input: Tensor<B, 3>,
}

impl<B> Mix6ForwardOutput<B>
where
    B: Backend,
{
    pub(crate) fn to_primitive(&self) -> Mix6ForwardPrimitiveOutput<B> {
        Mix6ForwardPrimitiveOutput {
            receptance_input: self.receptance_input.clone().into_primitive().tensor(),
            weight_decay_input: self.weight_decay_input.clone().into_primitive().tensor(),
            key_input: self.key_input.clone().into_primitive().tensor(),
            value_input: self.value_input.clone().into_primitive().tensor(),
            learning_rate_input: self.learning_rate_input.clone().into_primitive().tensor(),
            gate_input: self.gate_input.clone().into_primitive().tensor(),
        }
    }
}

/// Primitive outputs produced by the fused mix6 operation.
#[derive(Clone, Debug)]
pub struct Mix6ForwardPrimitiveOutput<B: Backend> {
    /// Primitive receptance branch input shaped `[batch_size, context_len, embedded_dim]`.
    pub receptance_input: FloatTensor<B>,
    /// Primitive weight-decay branch input shaped `[batch_size, context_len, embedded_dim]`.
    pub weight_decay_input: FloatTensor<B>,
    /// Primitive key branch input shaped `[batch_size, context_len, embedded_dim]`.
    pub key_input: FloatTensor<B>,
    /// Primitive value branch input shaped `[batch_size, context_len, embedded_dim]`.
    pub value_input: FloatTensor<B>,
    /// Primitive learning-rate branch input shaped `[batch_size, context_len, embedded_dim]`.
    pub learning_rate_input: FloatTensor<B>,
    /// Primitive gate branch input shaped `[batch_size, context_len, embedded_dim]`.
    pub gate_input: FloatTensor<B>,
}
