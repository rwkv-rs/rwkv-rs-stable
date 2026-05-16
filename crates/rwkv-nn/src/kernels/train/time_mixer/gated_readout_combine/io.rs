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
    train::time_mixer::gated_readout_combine::GatedReadoutCombineBackend as Backend,
};

/// Public tensor inputs for the gated-readout combine operation.
#[derive(Debug, Clone)]
pub struct GatedReadoutCombineForwardInputs<B: Backend> {
    /// Raw WKV output shaped `[batch_size, context_len, num_heads, head_size]`.
    pub wkv_output: Tensor<B, 4>,
    /// GroupNorm affine scale shaped `[embedded_dim]`.
    pub norm_gamma: Tensor<B, 1>,
    /// GroupNorm affine bias shaped `[embedded_dim]`.
    pub norm_beta: Tensor<B, 1>,
    /// GroupNorm epsilon.
    pub norm_epsilon: f64,
    /// Output gate shaped `[batch_size, context_len, embedded_dim]`.
    pub gate: Tensor<B, 3>,
    /// WKV receptance input shaped `[batch_size, context_len, num_heads, head_size]`.
    pub receptance: Tensor<B, 4>,
    /// WKV replacement key input shaped `[batch_size, context_len, num_heads, head_size]`.
    pub replacement_key: Tensor<B, 4>,
    /// WKV value input shaped `[batch_size, context_len, num_heads, head_size]`.
    pub value: Tensor<B, 4>,
    /// Per-head receptance/key bonus shaped `[num_heads, head_size]`.
    pub bonus: Tensor<B, 2>,
}

impl<B> GatedReadoutCombineForwardInputs<B>
where
    B: Backend,
{
    pub(crate) fn to_primitive(&self) -> GatedReadoutCombineForwardPrimitiveInputs<B> {
        GatedReadoutCombineForwardPrimitiveInputs {
            wkv_output: self.wkv_output.clone().into_primitive().tensor(),
            norm_gamma: self.norm_gamma.clone().into_primitive().tensor(),
            norm_beta: self.norm_beta.clone().into_primitive().tensor(),
            norm_epsilon: self.norm_epsilon,
            gate: self.gate.clone().into_primitive().tensor(),
            receptance: self.receptance.clone().into_primitive().tensor(),
            replacement_key: self.replacement_key.clone().into_primitive().tensor(),
            value: self.value.clone().into_primitive().tensor(),
            bonus: self.bonus.clone().into_primitive().tensor(),
        }
    }

    pub(crate) fn check(&self) -> Result<(), KernelInputsError<B>> {
        let wkv_output = get_tensor_info("wkv_output", &self.wkv_output);
        let norm_gamma = get_tensor_info("norm_gamma", &self.norm_gamma);
        let norm_beta = get_tensor_info("norm_beta", &self.norm_beta);
        let gate = get_tensor_info("gate", &self.gate);
        let receptance = get_tensor_info("receptance", &self.receptance);
        let replacement_key = get_tensor_info("replacement_key", &self.replacement_key);
        let value = get_tensor_info("value", &self.value);
        let bonus = get_tensor_info("bonus", &self.bonus);

        let batch_size = wkv_output.dim(0);
        let context_len = wkv_output.dim(1);
        let num_heads = wkv_output.dim(2);
        let head_size = wkv_output.dim(3);
        let embedded_dim = num_heads * head_size;

        check_axis_non_empty(wkv_output.axis(3))?;
        check_shape(&norm_gamma, [embedded_dim])?;
        check_shape(&norm_beta, [embedded_dim])?;
        check_shape(&gate, [batch_size, context_len, embedded_dim])?;
        check_shape(
            &replacement_key,
            [batch_size, context_len, num_heads, head_size],
        )?;
        check_shape(&value, [batch_size, context_len, num_heads, head_size])?;
        check_shape(&receptance, [batch_size, context_len, num_heads, head_size])?;
        check_shape(&bonus, [num_heads, head_size])?;
        check_shape(&wkv_output, [batch_size, context_len, num_heads, head_size])?;
        check_same_dtype(&[
            &wkv_output,
            &norm_gamma,
            &norm_beta,
            &gate,
            &receptance,
            &replacement_key,
            &value,
            &bonus,
        ])?;
        check_same_device(&[
            &wkv_output,
            &norm_gamma,
            &norm_beta,
            &gate,
            &receptance,
            &replacement_key,
            &value,
            &bonus,
        ])?;

        Ok(())
    }
}

/// Primitive tensor inputs passed to the fused gated-readout combine kernel.
#[derive(Debug, Clone)]
pub struct GatedReadoutCombineForwardPrimitiveInputs<B: Backend> {
    /// Primitive raw WKV output shaped `[batch_size, context_len, num_heads, head_size]`.
    pub wkv_output: FloatTensor<B>,
    /// Primitive GroupNorm affine scale shaped `[embedded_dim]`.
    pub norm_gamma: FloatTensor<B>,
    /// Primitive GroupNorm affine bias shaped `[embedded_dim]`.
    pub norm_beta: FloatTensor<B>,
    /// GroupNorm epsilon.
    pub norm_epsilon: f64,
    /// Primitive output gate tensor shaped `[batch_size, context_len, embedded_dim]`.
    pub gate: FloatTensor<B>,
    /// Primitive receptance tensor shaped `[batch_size, context_len, num_heads, head_size]`.
    pub receptance: FloatTensor<B>,
    /// Primitive replacement key tensor shaped `[batch_size, context_len, num_heads, head_size]`.
    pub replacement_key: FloatTensor<B>,
    /// Primitive value tensor shaped `[batch_size, context_len, num_heads, head_size]`.
    pub value: FloatTensor<B>,
    /// Primitive per-head bonus tensor shaped `[num_heads, head_size]`.
    pub bonus: FloatTensor<B>,
}
