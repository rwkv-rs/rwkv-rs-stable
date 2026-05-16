mod backward;
mod forward;
/// Input and output containers for the RWKV7 key-prepare kernel.
pub mod io;
mod kernel;

use burn::tensor::{Tensor, TensorPrimitive};
use burn_cubecl::{
    CubeBackend,
    CubeElement,
    CubeRuntime,
    FloatElement,
    IntElement,
    element::BoolElement,
};

use crate::{
    functions::normalize::normalize,
    kernels::train::{
        layout::assert_linear_readable,
        time_mixer::key_prepare::io::{
            KeyPrepareForwardInputs,
            KeyPrepareForwardOutput,
            KeyPrepareForwardPrimitiveInputs,
            KeyPrepareForwardPrimitiveOutput,
        },
    },
};

/// Backend primitive capability for the fused RWKV7 key-prepare operation.
pub trait KeyPrepareBackend: burn::tensor::backend::Backend {
    /// Runs the RWKV7 key preparation primitive.
    fn fused_key_prepare(
        inputs: KeyPrepareForwardPrimitiveInputs<Self>,
    ) -> KeyPrepareForwardPrimitiveOutput<Self>;
}

/// Autodiff backend marker for the key-prepare kernel.
pub trait AutodiffBackend: KeyPrepareBackend + burn::tensor::backend::AutodiffBackend {}

impl<B> AutodiffBackend for B where B: KeyPrepareBackend + burn::tensor::backend::AutodiffBackend {}

impl<R, F, I, BT> KeyPrepareBackend for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
{
    fn fused_key_prepare(
        inputs: KeyPrepareForwardPrimitiveInputs<Self>,
    ) -> KeyPrepareForwardPrimitiveOutput<Self> {
        assert_linear_readable("key", &inputs.key);
        assert_linear_readable("key_removal", &inputs.key_removal);
        assert_linear_readable("learning_rate", &inputs.learning_rate);
        assert_linear_readable("key_replacement", &inputs.key_replacement);

        forward::fused_key_prepare::<R, F, I, BT>(inputs)
    }
}

/// Runs the fused RWKV7 key preparation path after validating the public input contract.
///
/// `key` and `learning_rate` must be contiguous and shaped
/// `[batch_size, context_len, embedded_dim]`. `key_removal` and `key_replacement` must be
/// contiguous and shaped `[embedded_dim]`. `head_size` must divide `embedded_dim`.
///
/// Mathematically, this ports RWKV-LM v7 `tmix_kk_pre_bf16_v5` with repository terminology:
/// `key_removal` maps to `k_k`, `learning_rate` maps to `a`, and `key_replacement` maps to
/// `k_a`. The outputs are `replacement_key`, `removal_key_normalized`, and `replacement`.
pub fn key_prepare_custom<B: KeyPrepareBackend>(
    inputs: KeyPrepareForwardInputs<B>,
) -> KeyPrepareForwardOutput<B> {
    inputs.check().unwrap();
    let output = B::fused_key_prepare(inputs.to_primitive());

    KeyPrepareForwardOutput {
        replacement_key: Tensor::from_primitive(TensorPrimitive::Float(output.replacement_key)),
        removal_key_normalized: Tensor::from_primitive(TensorPrimitive::Float(
            output.removal_key_normalized,
        )),
        replacement: Tensor::from_primitive(TensorPrimitive::Float(output.replacement)),
    }
}

/// Computes RWKV7 key preparation with regular Burn tensor operations.
///
/// This is the semantic reference for correctness and autodiff tests. The custom path fuses the
/// two elementwise output expressions and the per-head key normalization into one CubeCL launch.
pub fn key_prepare_reference<B: KeyPrepareBackend>(
    inputs: KeyPrepareForwardInputs<B>,
) -> KeyPrepareForwardOutput<B> {
    let [batch_size, context_len, embedded_dim] = inputs.key.dims();
    let num_heads = embedded_dim / inputs.head_size;
    let key_removal = inputs
        .key_removal
        .unsqueeze_dim::<2>(0)
        .unsqueeze_dim::<3>(0);
    let key_replacement = inputs
        .key_replacement
        .unsqueeze_dim::<2>(0)
        .unsqueeze_dim::<3>(0);
    let kk = inputs.key.clone() * key_removal;
    let kk = normalize(
        kk.reshape([batch_size, context_len, num_heads, inputs.head_size]),
        2.0,
        -1,
        1.0e-12,
    )
    .reshape([batch_size, context_len, embedded_dim]);
    let replacement_key =
        inputs.key * (1.0 + (inputs.learning_rate.clone() - 1.0) * key_replacement);
    let removal_key_normalized = -kk.clone();
    let replacement = kk * inputs.learning_rate;

    KeyPrepareForwardOutput {
        replacement_key,
        removal_key_normalized,
        replacement,
    }
}

/// Convenience wrapper for the fused RWKV7 key preparation path.
pub fn key_prepare<B: KeyPrepareBackend>(
    key: Tensor<B, 3>,
    key_removal: Tensor<B, 1>,
    learning_rate: Tensor<B, 3>,
    key_replacement: Tensor<B, 1>,
    head_size: usize,
) -> KeyPrepareForwardOutput<B> {
    key_prepare_custom(KeyPrepareForwardInputs {
        key,
        key_removal,
        learning_rate,
        key_replacement,
        head_size,
    })
}
