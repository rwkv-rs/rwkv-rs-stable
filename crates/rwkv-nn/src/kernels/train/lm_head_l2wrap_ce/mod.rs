mod backward;
mod forward;
mod io;
mod kernel;

use burn::{
    prelude::{Backend, Int},
    tensor::{FloatDType, Tensor, activation::log_softmax, ops::FloatTensor},
};
#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal",
    feature = "wgpu",
    feature = "webgpu"
))]
use burn::tensor::TensorPrimitive;
use burn_cubecl::{
    CubeBackend,
    CubeElement,
    CubeRuntime,
    FloatElement,
    IntElement,
    element::BoolElement,
};

use crate::kernels::{
    check::{
        KernelInputsError,
        check_axes_equal,
        check_axis_non_empty,
        check_same_device,
        get_tensor_info,
    },
    train::{layout::assert_linear_readable, lm_head_l2wrap_ce::io::LmHeadL2WrapCePrimitiveInputs},
};

const L2WRAP_FACTOR: f64 = 1e-4;

/// Backend primitive capability for LM-head cross entropy with RWKV-LM L2Wrap gradients.
pub trait LmHeadL2WrapCeBackend: Backend {
    /// Runs the fused LM-head L2Wrap cross-entropy primitive.
    fn fused_lm_head_l2wrap_ce(inputs: LmHeadL2WrapCePrimitiveInputs<Self>) -> FloatTensor<Self>;
}

/// Autodiff backend marker for the LM-head L2Wrap cross-entropy kernel.
pub trait AutodiffBackend: LmHeadL2WrapCeBackend + burn::tensor::backend::AutodiffBackend {}

impl<B> AutodiffBackend for B where B: LmHeadL2WrapCeBackend + burn::tensor::backend::AutodiffBackend
{}

impl<R, F, I, BT> LmHeadL2WrapCeBackend for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
{
    fn fused_lm_head_l2wrap_ce(inputs: LmHeadL2WrapCePrimitiveInputs<Self>) -> FloatTensor<Self> {
        assert_linear_readable("logits", &inputs.logits);
        assert_linear_readable("targets", &inputs.targets);

        forward::fused_lm_head_l2wrap_ce::<R, F, I, BT>(inputs)
    }
}

/// Runs LM-head cross entropy with RWKV-LM L2Wrap gradient semantics.
///
/// `logits` must be shaped `[batch_size, context_len, vocab_size]`.
/// `targets` must be shaped `[batch_size, context_len]` and contain valid token ids in the
/// vocabulary range. The returned tensor is shaped `[1]`, matching Burn loss metric inputs and
/// `NextTokenPredictionOutput`.
///
/// The forward value is standard mean next-token cross entropy. During backward, each
/// `[batch, time]` row receives the RWKV-LM L2Wrap extra gradient at its argmax logit:
/// `grad[row, argmax] += max_logit * 1e-4 / (batch_size * context_len)`.
pub fn lm_head_l2wrap_ce<B: LmHeadL2WrapCeBackend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
) -> Tensor<B, 1> {
    lm_head_l2wrap_ce_custom(logits, targets)
}

/// Runs the fused LM-head L2Wrap cross-entropy path after validating the public input contract.
///
/// Cube backends reduce each `[batch, time]` row directly to the scalar mean cross entropy without
/// materializing one-hot targets or full log-softmax tensors.
#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "metal",
    feature = "wgpu",
    feature = "webgpu"
))]
pub fn lm_head_l2wrap_ce_custom<B: LmHeadL2WrapCeBackend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
) -> Tensor<B, 1> {
    check_inputs(&logits, &targets).unwrap();

    let output = B::fused_lm_head_l2wrap_ce(LmHeadL2WrapCePrimitiveInputs {
        logits: logits.into_primitive().tensor(),
        targets: targets.into_primitive(),
    });

    Tensor::from_primitive(TensorPrimitive::Float(output))
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
pub fn lm_head_l2wrap_ce_custom<B: Backend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
) -> Tensor<B, 1> {
    lm_head_l2wrap_ce_reference(logits, targets)
}

/// Burn-ops reference for LM-head cross entropy with L2Wrap backward behavior.
///
/// The `l2wrap - l2wrap.detach()` term preserves the CE-only forward value while retaining the
/// exact additional backward gradient from the old RWKV-LM L2Wrap implementation.
pub fn lm_head_l2wrap_ce_reference<B: Backend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
) -> Tensor<B, 1> {
    check_inputs(&logits, &targets).unwrap();

    let [batch_size, context_len, vocab_size] = logits.dims();
    let num_tokens = batch_size * context_len;
    let logits_2d = logits.clone().reshape([num_tokens, vocab_size]);
    let device = logits.device();
    let target_ids = targets.reshape([num_tokens, 1]);
    let vocab_ids =
        Tensor::<B, 1, Int>::arange(0..vocab_size as i64, &device).reshape([1, vocab_size]);
    let target_probs = target_ids.equal(vocab_ids).cast(FloatDType::F32);

    let cross_entropy = (log_softmax(logits_2d, 1) * target_probs)
        .sum_dim(1)
        .mean()
        .neg();
    let max_logits = logits.max_dim(2);
    let l2wrap = max_logits.powf_scalar(2.0).sum() * (0.5 * L2WRAP_FACTOR / num_tokens as f64);

    cross_entropy + l2wrap.clone() - l2wrap.detach()
}

fn check_inputs<B: Backend>(
    logits: &Tensor<B, 3>,
    targets: &Tensor<B, 2, Int>,
) -> Result<(), KernelInputsError<B>> {
    let logits = get_tensor_info("logits", logits);
    let targets = get_tensor_info("targets", targets);

    check_axis_non_empty(logits.axis(0))?;
    check_axis_non_empty(logits.axis(1))?;
    check_axis_non_empty(logits.axis(2))?;
    check_axes_equal(&[logits.axis(0), targets.axis(0)])?;
    check_axes_equal(&[logits.axis(1), targets.axis(1)])?;
    check_same_device(&[&logits, &targets])?;

    Ok(())
}
