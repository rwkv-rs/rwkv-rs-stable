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
    train::time_mixer::key_prepare::KeyPrepareBackend as Backend,
};

/// Public tensor inputs for RWKV7 key preparation.
#[derive(Debug, Clone)]
pub struct KeyPrepareForwardInputs<B: Backend> {
    /// Key precursor shaped `[batch_size, context_len, embedded_dim]`.
    pub key: Tensor<B, 3>,
    /// Per-embedded-dimension key removal scale shaped `[embedded_dim]`.
    pub key_removal: Tensor<B, 1>,
    /// Learning-rate tensor shaped `[batch_size, context_len, embedded_dim]`.
    pub learning_rate: Tensor<B, 3>,
    /// Per-embedded-dimension key replacement scale shaped `[embedded_dim]`.
    pub key_replacement: Tensor<B, 1>,
    /// Per-head channel count used by the L2 normalization.
    pub head_size: usize,
}

impl<B> KeyPrepareForwardInputs<B>
where
    B: Backend,
{
    pub(crate) fn to_primitive(&self) -> KeyPrepareForwardPrimitiveInputs<B> {
        KeyPrepareForwardPrimitiveInputs {
            key: self.key.clone().into_primitive().tensor(),
            key_removal: self.key_removal.clone().into_primitive().tensor(),
            learning_rate: self.learning_rate.clone().into_primitive().tensor(),
            key_replacement: self.key_replacement.clone().into_primitive().tensor(),
            head_size: self.head_size,
        }
    }

    pub(crate) fn check(&self) -> Result<(), KernelInputsError<B>> {
        let key = get_tensor_info("key", &self.key);
        let key_removal = get_tensor_info("key_removal", &self.key_removal);
        let learning_rate = get_tensor_info("learning_rate", &self.learning_rate);
        let key_replacement = get_tensor_info("key_replacement", &self.key_replacement);
        let embedded_dim = key.dim(2);

        assert!(self.head_size > 0, "head_size must be non-zero");
        assert!(
            embedded_dim.is_multiple_of(self.head_size),
            "embedded_dim must be a multiple of head_size"
        );
        check_axis_non_empty(key.axis(2))?;
        check_same_shape(&[&key, &learning_rate])?;
        check_shape(&key_removal, [embedded_dim])?;
        check_shape(&key_replacement, [embedded_dim])?;
        check_same_dtype(&[&key, &key_removal, &learning_rate, &key_replacement])?;
        check_same_device(&[&key, &key_removal, &learning_rate, &key_replacement])?;

        Ok(())
    }
}

/// Primitive tensor inputs passed to the fused key-prepare kernel.
#[derive(Debug, Clone)]
pub struct KeyPrepareForwardPrimitiveInputs<B: Backend> {
    /// Primitive key precursor.
    pub key: FloatTensor<B>,
    /// Primitive key removal scale.
    pub key_removal: FloatTensor<B>,
    /// Primitive learning-rate tensor.
    pub learning_rate: FloatTensor<B>,
    /// Primitive key replacement scale.
    pub key_replacement: FloatTensor<B>,
    /// Per-head channel count used by the L2 normalization.
    pub head_size: usize,
}

/// Public tensor outputs for RWKV7 key preparation.
#[derive(Debug, Clone)]
pub struct KeyPrepareForwardOutput<B: Backend> {
    /// Replacement key shaped `[batch_size, context_len, embedded_dim]`.
    pub replacement_key: Tensor<B, 3>,
    /// Normalized removal key shaped `[batch_size, context_len, embedded_dim]`.
    pub removal_key_normalized: Tensor<B, 3>,
    /// Replacement tensor shaped `[batch_size, context_len, embedded_dim]`.
    pub replacement: Tensor<B, 3>,
}

impl<B> KeyPrepareForwardOutput<B>
where
    B: Backend,
{
    #[cfg(not(any(
        feature = "cuda",
        feature = "rocm",
        feature = "vulkan",
        feature = "metal",
        feature = "wgpu",
        feature = "webgpu"
    )))]
    pub(crate) fn to_primitive(&self) -> KeyPrepareForwardPrimitiveOutput<B> {
        KeyPrepareForwardPrimitiveOutput {
            replacement_key: self.replacement_key.clone().into_primitive().tensor(),
            removal_key_normalized: self
                .removal_key_normalized
                .clone()
                .into_primitive()
                .tensor(),
            replacement: self.replacement.clone().into_primitive().tensor(),
        }
    }
}

/// Primitive tensor outputs produced by the fused key-prepare kernel.
#[derive(Debug, Clone)]
pub struct KeyPrepareForwardPrimitiveOutput<B: Backend> {
    /// Primitive replacement key.
    pub replacement_key: FloatTensor<B>,
    /// Primitive normalized removal key.
    pub removal_key_normalized: FloatTensor<B>,
    /// Primitive replacement tensor.
    pub replacement: FloatTensor<B>,
}
