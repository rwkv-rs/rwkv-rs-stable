mod backward;
mod forward;
/// Input and output containers for the learning-rate gate kernel.
pub mod io;
mod kernel;

use burn::tensor::{Tensor, TensorPrimitive, activation::sigmoid, ops::FloatTensor};
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
    time_mixer::learning_rate_gate::io::{
        LearningRateGateForwardInputs,
        LearningRateGateForwardPrimitiveInputs,
    },
};

/// Backend primitive capability for the fused learning-rate gate.
pub trait LearningRateGateBackend: burn::tensor::backend::Backend {
    /// Runs `sigmoid(learning_rate_base + learning_rate_input)` as one fused primitive operation.
    fn fused_learning_rate_gate(
        inputs: LearningRateGateForwardPrimitiveInputs<Self>,
    ) -> FloatTensor<Self>;
}

/// Autodiff backend marker for the learning-rate gate.
pub trait AutodiffBackend:
    LearningRateGateBackend + burn::tensor::backend::AutodiffBackend
{
}

impl<B> AutodiffBackend for B where
    B: LearningRateGateBackend + burn::tensor::backend::AutodiffBackend
{
}

impl<R, F, I, BT> LearningRateGateBackend for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
{
    fn fused_learning_rate_gate(
        inputs: LearningRateGateForwardPrimitiveInputs<Self>,
    ) -> FloatTensor<Self> {
        assert_linear_readable("learning_rate_base", &inputs.learning_rate_base);
        assert_linear_readable("learning_rate_input", &inputs.learning_rate_input);

        forward::fused_learning_rate_gate::<R, F, I, BT>(inputs)
    }
}

/// Runs the fused RWKV7 learning-rate gate after validating the public input contract.
///
/// `learning_rate_base` must be contiguous and shaped `[embedded_dim]`.
/// `learning_rate_input` must be contiguous and shaped
/// `[batch_size, context_len, embedded_dim]`.
///
/// Mathematically:
/// `learning_rate[b, t, e] = sigmoid(learning_rate_base[e] + learning_rate_input[b, t, e])`.
/// This ports the RWKV-LM `a_gate` fast path using repository terminology: the CUDA fixture name
/// maps `a0` to `learning_rate_base` and `a12` to `learning_rate_input`. The custom path fuses the
/// broadcast add and sigmoid into one elementwise kernel and keeps the embedded-dimension
/// broadcast as index math.
pub fn learning_rate_gate_custom<B: LearningRateGateBackend>(
    inputs: LearningRateGateForwardInputs<B>,
) -> Tensor<B, 3> {
    inputs.check().unwrap();
    let output = B::fused_learning_rate_gate(inputs.to_primitive());

    Tensor::from_primitive(TensorPrimitive::Float(output))
}

/// Computes the learning-rate gate with regular Burn tensor operations.
///
/// This is the semantic reference for correctness and gradient tests. Burn fusion may simplify the
/// expression graph for `sigmoid(learning_rate_base + learning_rate_input)`, while the custom path
/// expresses the intended fused primitive and exposes CubeCL autotune choices for vector width.
pub fn learning_rate_gate_reference<B: LearningRateGateBackend>(
    inputs: LearningRateGateForwardInputs<B>,
) -> Tensor<B, 3> {
    sigmoid(
        inputs.learning_rate_input
            + inputs
                .learning_rate_base
                .unsqueeze_dim::<2>(0)
                .unsqueeze_dim::<3>(0),
    )
}

/// Convenience wrapper for the fused learning-rate gate path.
pub fn learning_rate_gate<B: LearningRateGateBackend>(
    learning_rate_base: Tensor<B, 1>,
    learning_rate_input: Tensor<B, 3>,
) -> Tensor<B, 3> {
    learning_rate_gate_custom(LearningRateGateForwardInputs {
        learning_rate_base,
        learning_rate_input,
    })
}
