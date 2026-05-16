mod backward;
mod forward;
/// Input and output containers for the RWKV7 pretrain mix6 kernel.
pub mod io;
mod kernel;

use burn::tensor::{Tensor, TensorPrimitive, ops::FloatTensor};
use burn_cubecl::{
    CubeBackend,
    CubeElement,
    CubeRuntime,
    FloatElement,
    IntElement,
    element::BoolElement,
};

use crate::kernels::train::{
    layout::assert_linear_readable,
    time_mixer::mix6::io::{
        Mix6ForwardInputs,
        Mix6ForwardOutput,
        Mix6ForwardPrimitiveInputs,
        Mix6ForwardPrimitiveOutput,
    },
};

/// Backend primitive capability for the RWKV7 pretrain mix6 operation.
pub trait Mix6Backend: burn::tensor::backend::Backend {
    /// Runs the six-output time-mix primitive.
    fn fused_mix6(inputs: Mix6ForwardPrimitiveInputs<Self>) -> Mix6ForwardPrimitiveOutput<Self>;
}

/// Autodiff backend marker for the mix6 kernel.
pub trait AutodiffBackend: Mix6Backend + burn::tensor::backend::AutodiffBackend {}

impl<B> AutodiffBackend for B where B: Mix6Backend + burn::tensor::backend::AutodiffBackend {}

#[doc(hidden)]
pub trait Mix6StackedBackend: Mix6Backend {
    fn fused_mix6_stacked(inputs: Mix6ForwardPrimitiveInputs<Self>) -> FloatTensor<Self>;
}

impl<R, F, I, BT> Mix6Backend for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
{
    fn fused_mix6(inputs: Mix6ForwardPrimitiveInputs<Self>) -> Mix6ForwardPrimitiveOutput<Self> {
        assert_linear_readable("embedded_context", &inputs.embedded_context);
        assert_linear_readable("receptance_scale", &inputs.receptance_scale);
        assert_linear_readable("weight_decay_scale", &inputs.weight_decay_scale);
        assert_linear_readable("key_scale", &inputs.key_scale);
        assert_linear_readable("value_scale", &inputs.value_scale);
        assert_linear_readable("learning_rate_scale", &inputs.learning_rate_scale);
        assert_linear_readable("gate_scale", &inputs.gate_scale);

        forward::fused_mix6::<R, F, I, BT>(inputs)
    }
}

/// Runs the fused RWKV7 pretrain time-mix kernel.
///
/// `embedded_context` must be shaped `[batch_size, context_len, embedded_dim]`.
/// Each scale tensor must be shaped `[1, 1, embedded_dim]`.
///
/// For every token position the operation first computes the pretrain token-shift difference:
/// `previous = 0` for `time_index == 0`, otherwise the previous token from the same batch;
/// `token_shifted_diff = previous - current`. Each branch then computes
/// `current + token_shifted_diff * branch_scale`.
///
/// The custom path fuses the shift difference and six branch add/multiply expressions into one
/// multi-output kernel. It shares the current token, previous token, and difference values across
/// the six branch writes while keeping each scale broadcast as embedded-dimension index math.
#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal",
    feature = "wgpu",
    feature = "webgpu"
))]
pub fn mix6_custom<B: Mix6Backend>(inputs: Mix6ForwardInputs<B>) -> Mix6ForwardOutput<B> {
    inputs.check().unwrap();
    let output = B::fused_mix6(inputs.to_primitive());

    Mix6ForwardOutput {
        receptance_input: Tensor::from_primitive(TensorPrimitive::Float(output.receptance_input)),
        weight_decay_input: Tensor::from_primitive(TensorPrimitive::Float(
            output.weight_decay_input,
        )),
        key_input: Tensor::from_primitive(TensorPrimitive::Float(output.key_input)),
        value_input: Tensor::from_primitive(TensorPrimitive::Float(output.value_input)),
        learning_rate_input: Tensor::from_primitive(TensorPrimitive::Float(
            output.learning_rate_input,
        )),
        gate_input: Tensor::from_primitive(TensorPrimitive::Float(output.gate_input)),
    }
}

#[cfg(not(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal",
    feature = "wgpu",
    feature = "webgpu"
)))]
/// CPU-only fallback that keeps unit tests on the Burn reference semantics.
pub fn mix6_custom<B: Mix6Backend>(inputs: Mix6ForwardInputs<B>) -> Mix6ForwardOutput<B> {
    inputs.check().unwrap();
    mix6_reference(inputs)
}

#[doc(hidden)]
pub fn mix6_stacked_custom<B: Mix6StackedBackend>(inputs: Mix6ForwardInputs<B>) -> Tensor<B, 4> {
    inputs.check().unwrap();
    Tensor::from_primitive(TensorPrimitive::Float(B::fused_mix6_stacked(
        inputs.to_primitive(),
    )))
}

/// Computes RWKV7 pretrain mix6 with regular Burn tensor operations.
///
/// This is the semantic reference for correctness and gradient tests. Burn fusion may simplify
/// parts of the generic graph, while the custom path exposes explicit cross-output sharing.
pub fn mix6_reference<B: Mix6Backend>(inputs: Mix6ForwardInputs<B>) -> Mix6ForwardOutput<B> {
    let [batch_size, context_len, embedded_dim] = inputs.embedded_context.dims();

    if context_len == 0 {
        return Mix6ForwardOutput {
            receptance_input: inputs.embedded_context.clone(),
            weight_decay_input: inputs.embedded_context.clone(),
            key_input: inputs.embedded_context.clone(),
            value_input: inputs.embedded_context.clone(),
            learning_rate_input: inputs.embedded_context.clone(),
            gate_input: inputs.embedded_context,
        };
    }

    let device = inputs.embedded_context.device();
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

    Mix6ForwardOutput {
        receptance_input: inputs.embedded_context.clone()
            + token_shifted_diff.clone() * inputs.receptance_scale,
        weight_decay_input: inputs.embedded_context.clone()
            + token_shifted_diff.clone() * inputs.weight_decay_scale,
        key_input: inputs.embedded_context.clone() + token_shifted_diff.clone() * inputs.key_scale,
        value_input: inputs.embedded_context.clone()
            + token_shifted_diff.clone() * inputs.value_scale,
        learning_rate_input: inputs.embedded_context.clone()
            + token_shifted_diff.clone() * inputs.learning_rate_scale,
        gate_input: inputs.embedded_context + token_shifted_diff * inputs.gate_scale,
    }
}

/// Convenience wrapper for the fused pretrain mix6 path.
pub fn mix6<B: Mix6Backend>(
    embedded_context: Tensor<B, 3>,
    receptance_scale: Tensor<B, 3>,
    weight_decay_scale: Tensor<B, 3>,
    key_scale: Tensor<B, 3>,
    value_scale: Tensor<B, 3>,
    learning_rate_scale: Tensor<B, 3>,
    gate_scale: Tensor<B, 3>,
) -> Mix6ForwardOutput<B> {
    mix6_custom(Mix6ForwardInputs {
        embedded_context,
        receptance_scale,
        weight_decay_scale,
        key_scale,
        value_scale,
        learning_rate_scale,
        gate_scale,
    })
}
