mod backward;
mod forward;
/// Input containers for the RWKV7 pretrain channel mixer kernel.
pub mod io;
mod kernel;

use burn::tensor::{Tensor, TensorPrimitive, activation::relu, ops::FloatTensor};
use burn_cubecl::{
    CubeBackend,
    CubeElement,
    CubeRuntime,
    FloatElement,
    IntElement,
    element::BoolElement,
};

use crate::kernels::train::{
    channel_mixer::io::{ChannelMixerForwardInputs, ChannelMixerForwardPrimitiveInputs},
    layout::assert_linear_readable,
};

/// Backend primitive capability for the fused RWKV7 pretrain channel mixer.
pub trait ChannelMixerBackend: burn::tensor::backend::Backend {
    /// Runs the channel mixer layer as one primitive operation.
    fn fused_channel_mixer(inputs: ChannelMixerForwardPrimitiveInputs<Self>) -> FloatTensor<Self>;
}

/// Autodiff backend marker for the channel mixer kernel.
pub trait AutodiffBackend: ChannelMixerBackend + burn::tensor::backend::AutodiffBackend {}

impl<B> AutodiffBackend for B where B: ChannelMixerBackend + burn::tensor::backend::AutodiffBackend {}

impl<R, F, I, BT> ChannelMixerBackend for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
{
    fn fused_channel_mixer(inputs: ChannelMixerForwardPrimitiveInputs<Self>) -> FloatTensor<Self> {
        assert_linear_readable("embedded_context", &inputs.embedded_context);
        assert_linear_readable("key_scale", &inputs.key_scale);
        assert_linear_readable("key_weight", &inputs.key_weight);
        assert_linear_readable("value_weight", &inputs.value_weight);

        forward::fused_channel_mixer::<R, F, I, BT>(inputs)
    }
}

/// Runs the fused RWKV7 pretrain channel mixer after validating the public input contract.
///
/// `embedded_context` must be contiguous and shaped `[batch_size, context_len, embedded_dim]`.
/// `key_scale` must be contiguous and shaped `[embedded_dim]`. `key_weight` must be contiguous
/// and shaped `[embedded_dim, expanded_dim]`. `value_weight` must be contiguous and shaped
/// `[expanded_dim, embedded_dim]`.
///
/// For each token, the operation first computes the token-shift channel mix:
/// `previous = 0` for `time_index == 0`, otherwise the previous token from the same batch;
/// `key_input = embedded_context + (previous - embedded_context) * key_scale`.
/// It then computes `activated_key = relu(key_input @ key_weight)^2` and
/// `output = activated_key @ value_weight`.
///
/// This ports the RWKV-LM `cmix` fast path using repository terminology. The custom primitive
/// fuses the token-shift mix and ReLU-square elementwise stages while using backend matmul
/// primitives for the two projections.
pub fn channel_mixer_custom<B: ChannelMixerBackend>(
    inputs: ChannelMixerForwardInputs<B>,
) -> Tensor<B, 3> {
    inputs.check().unwrap();
    let output = B::fused_channel_mixer(inputs.to_primitive());

    Tensor::from_primitive(TensorPrimitive::Float(output))
}

/// Computes the RWKV7 pretrain channel mixer with regular Burn tensor operations.
///
/// This is the semantic reference for correctness and gradient tests. Burn fusion may simplify
/// parts of the expression graph, while the custom path exposes explicit fused elementwise stages
/// around the backend matmul operations.
pub fn channel_mixer_reference<B: burn::tensor::backend::Backend>(
    inputs: ChannelMixerForwardInputs<B>,
) -> Tensor<B, 3> {
    let [batch_size, context_len, embedded_dim] = inputs.embedded_context.dims();
    let device = inputs.embedded_context.device();

    if context_len == 0 {
        return Tensor::<B, 3>::zeros([batch_size, 0, embedded_dim], &device);
    }

    let zero = Tensor::<B, 3>::zeros([batch_size, 1, embedded_dim], &device);
    let shifted = if context_len == 1 {
        zero
    } else {
        Tensor::cat(
            vec![
                zero,
                inputs.embedded_context.clone().slice([
                    0..batch_size,
                    0..(context_len - 1),
                    0..embedded_dim,
                ]),
            ],
            1,
        )
    };
    let token_shifted_diff = shifted - inputs.embedded_context.clone();
    let key_input = inputs.embedded_context
        + token_shifted_diff * inputs.key_scale.unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0);
    let rows = batch_size * context_len;
    let key_input = key_input.reshape([rows, embedded_dim]);
    let activated_key = relu(key_input.matmul(inputs.key_weight)).powf_scalar(2.0);
    let output = activated_key.matmul(inputs.value_weight);

    output.reshape([batch_size, context_len, embedded_dim])
}

/// Convenience wrapper for the fused pretrain channel mixer path.
pub fn channel_mixer<B: ChannelMixerBackend>(
    embedded_context: Tensor<B, 3>,
    key_scale: Tensor<B, 1>,
    key_weight: Tensor<B, 2>,
    value_weight: Tensor<B, 2>,
) -> Tensor<B, 3> {
    channel_mixer_custom(ChannelMixerForwardInputs {
        embedded_context,
        key_scale,
        key_weight,
        value_weight,
    })
}
