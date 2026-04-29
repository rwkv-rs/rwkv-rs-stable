use burn::tensor::{Tensor, ops::FloatTensor};

use crate::kernels::{
    check::{
        KernelInputsError,
        check_axes_equal,
        check_axis_non_empty,
        check_same_device,
        check_same_dtype,
        check_shape,
        get_tensor_info,
    },
    train::channel_mixer::ChannelMixerBackend as Backend,
};

/// Public tensor inputs for the RWKV7 pretrain channel mixer.
#[derive(Debug, Clone)]
pub struct ChannelMixerForwardInputs<B: burn::tensor::backend::Backend> {
    /// Input context tensor shaped `[batch_size, context_len, embedded_dim]`.
    pub embedded_context: Tensor<B, 3>,
    /// Per-embedded-dimension token-shift mix scale shaped `[embedded_dim]`.
    pub key_scale: Tensor<B, 1>,
    /// Key projection weight shaped `[expanded_dim, embedded_dim]`.
    pub key_weight: Tensor<B, 2>,
    /// Value projection weight shaped `[embedded_dim, expanded_dim]`.
    pub value_weight: Tensor<B, 2>,
}

impl<B> ChannelMixerForwardInputs<B>
where
    B: Backend,
{
    pub(crate) fn to_primitive(&self) -> ChannelMixerForwardPrimitiveInputs<B> {
        ChannelMixerForwardPrimitiveInputs {
            embedded_context: self.embedded_context.clone().into_primitive().tensor(),
            key_scale: self.key_scale.clone().into_primitive().tensor(),
            key_weight: self.key_weight.clone().into_primitive().tensor(),
            value_weight: self.value_weight.clone().into_primitive().tensor(),
        }
    }
}

impl<B> ChannelMixerForwardInputs<B>
where
    B: Backend,
{
    pub(crate) fn check(&self) -> Result<(), KernelInputsError<B>> {
        let embedded_context = get_tensor_info("embedded_context", &self.embedded_context);
        let key_scale = get_tensor_info("key_scale", &self.key_scale);
        let key_weight = get_tensor_info("key_weight", &self.key_weight);
        let value_weight = get_tensor_info("value_weight", &self.value_weight);
        let embedded_dim = embedded_context.dim(2);
        let expanded_dim = key_weight.dim(0);

        check_axis_non_empty(embedded_context.axis(2))?;
        check_axis_non_empty(key_weight.axis(0))?;
        check_shape(&key_scale, [embedded_dim])?;
        check_axes_equal(&[
            embedded_context.axis(2),
            key_weight.axis(1),
            value_weight.axis(0),
        ])?;
        check_axes_equal(&[key_weight.axis(0), value_weight.axis(1)])?;
        check_shape(&value_weight, [embedded_dim, expanded_dim])?;
        check_same_dtype(&[&embedded_context, &key_scale, &key_weight, &value_weight])?;
        check_same_device(&[&embedded_context, &key_scale, &key_weight, &value_weight])?;

        Ok(())
    }
}

/// Primitive tensor inputs passed to the fused channel mixer kernel.
#[derive(Debug, Clone)]
pub struct ChannelMixerForwardPrimitiveInputs<B: Backend> {
    /// Primitive context tensor shaped `[batch_size, context_len, embedded_dim]`.
    pub embedded_context: FloatTensor<B>,
    /// Primitive mix scale shaped `[embedded_dim]`.
    pub key_scale: FloatTensor<B>,
    /// Primitive key projection weight shaped `[expanded_dim, embedded_dim]`.
    pub key_weight: FloatTensor<B>,
    /// Primitive value projection weight shaped `[embedded_dim, expanded_dim]`.
    pub value_weight: FloatTensor<B>,
}

/// Primitive gradients produced by the fused channel mixer backward pass.
#[derive(Debug, Clone)]
pub struct ChannelMixerBackwardPrimitiveOutputs<B: Backend> {
    /// Gradient for `embedded_context`, shaped `[batch_size, context_len, embedded_dim]`.
    pub embedded_context_grad: FloatTensor<B>,
    /// Gradient for `key_scale`, shaped `[embedded_dim]`.
    pub key_scale_grad: FloatTensor<B>,
    /// Gradient for `key_weight`, shaped `[expanded_dim, embedded_dim]`.
    pub key_weight_grad: FloatTensor<B>,
    /// Gradient for `value_weight`, shaped `[embedded_dim, expanded_dim]`.
    pub value_weight_grad: FloatTensor<B>,
}
