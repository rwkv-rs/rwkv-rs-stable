pub(crate) mod backward;
pub(crate) mod forward;
/// Input and output containers for addcmul kernels.
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

use crate::kernels::{
    Backend,
    addcmul::io::{
        Addcmul5ForwardInputs,
        Addcmul5ForwardOutput,
        Addcmul5ForwardPrimitiveInputs,
        Addcmul5ForwardPrimitiveOutput,
        AddcmulForwardInputs,
        AddcmulForwardPrimitiveInputs,
    },
};

/// We create our own Backend trait that extends the Burn backend trait.
pub trait AddcmulBackend: burn::tensor::backend::Backend {
    /// Please name this function "fused_{kernel_name}" when you reuse this template.
    fn fused_addcmul(inputs: AddcmulForwardPrimitiveInputs<Self>) -> FloatTensor<Self>;

    /// Runs the fused five-output addcmul primitive.
    fn fused_addcmul5(
        inputs: Addcmul5ForwardPrimitiveInputs<Self>,
    ) -> Addcmul5ForwardPrimitiveOutput<Self>;
}

/// We create our own AutodiffBackend trait that extends the Burn autodiff backend trait.
pub trait AutodiffBackend: AddcmulBackend + burn::tensor::backend::AutodiffBackend {}

impl<B> AutodiffBackend for B where B: AddcmulBackend + burn::tensor::backend::AutodiffBackend {}

impl<R, F, I, BT> AddcmulBackend for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
{
    fn fused_addcmul(inputs: AddcmulForwardPrimitiveInputs<Self>) -> FloatTensor<Self> {
        assert!(inputs.base.is_contiguous(), "base must be contiguous");
        assert!(inputs.diff.is_contiguous(), "diff must be contiguous");
        assert!(inputs.scale.is_contiguous(), "scale must be contiguous");

        forward::fused_addcmul::<R, F, I, BT>(inputs)
    }

    fn fused_addcmul5(
        inputs: Addcmul5ForwardPrimitiveInputs<Self>,
    ) -> Addcmul5ForwardPrimitiveOutput<Self> {
        assert!(inputs.base.is_contiguous(), "base must be contiguous");
        assert!(inputs.diff.is_contiguous(), "diff must be contiguous");
        assert!(
            inputs.receptance_scale.is_contiguous(),
            "receptance_scale must be contiguous"
        );
        assert!(
            inputs.weight_decay_scale.is_contiguous(),
            "weight_decay_scale must be contiguous"
        );
        assert!(
            inputs.key_scale.is_contiguous(),
            "key_scale must be contiguous"
        );
        assert!(
            inputs.value_scale.is_contiguous(),
            "value_scale must be contiguous"
        );
        assert!(
            inputs.learning_rate_scale.is_contiguous(),
            "learning_rate_scale must be contiguous"
        );

        forward::fused_addcmul5::<R, F, I, BT>(inputs)
    }
}

/// Runs the fused addcmul kernel after validating the public input contract.
///
/// `base` and `diff` must be contiguous tensors shaped
/// `[batch_size, context_len, embedded_dim]`.
/// `scale` must be contiguous and shaped `[1, 1, embedded_dim]`.
///
/// Mathematically, each output element is independent:
/// `output[b, t, e] = base[b, t, e] + diff[b, t, e] * scale[0, 0, e]`.
/// The custom path maps that equation directly to a single fused kernel. It avoids the generic
/// reference graph's separate multiply/add operations and keeps the scale broadcast as an index
/// calculation instead of materializing an expanded scale tensor. The kernel can reuse `base` as
/// the destination when Burn reports that the input storage is mutable and non-overlapping;
/// otherwise it writes one output tensor.
pub fn addcmul_custom<B: AddcmulBackend>(inputs: AddcmulForwardInputs<B>) -> Tensor<B, 3> {
    inputs.check().unwrap();
    let output = B::fused_addcmul(inputs.to_primitive());

    Tensor::from_primitive(TensorPrimitive::Float(output))
}

/// Computes addcmul with regular tensor operations as the semantic reference.
///
/// This returns `base + diff * scale` with Burn's normal broadcasting semantics.
/// Burn fusion and autotune may reduce temporary materialization for this expression, but the
/// reference still describes the work as generic tensor multiply and add nodes. It is kept as the
/// correctness baseline, not as the performance target.
pub fn addcmul_reference<B: Backend>(inputs: AddcmulForwardInputs<B>) -> Tensor<B, 3> {
    inputs.base + inputs.diff * inputs.scale
}

/// Runs the fused five-scale addcmul kernel after validating the public input contract.
///
/// `base` and `diff` must be contiguous tensors shaped
/// `[batch_size, context_len, embedded_dim]`.
/// Each scale tensor must be contiguous and shaped `[1, 1, embedded_dim]`.
///
/// Mathematically, the five outputs share the same `base[b, t, e]` and `diff[b, t, e]` values:
/// each branch computes `base[b, t, e] + diff[b, t, e] * branch_scale[0, 0, e]`.
/// The custom path evaluates the five branch equations in one multi-output kernel, so `base` and
/// `diff` are loaded once per vectorized position and reused for all branch writes. This removes
/// the reference path's repeated generic addcmul expressions; Burn may fuse those generic ops, but
/// it still cannot express the same explicit cross-output load sharing as this custom kernel.
pub fn addcmul5_custom<B: AddcmulBackend>(
    inputs: Addcmul5ForwardInputs<B>,
) -> Addcmul5ForwardOutput<B> {
    inputs.check().unwrap();
    let output = B::fused_addcmul5(inputs.to_primitive());

    Addcmul5ForwardOutput {
        receptance_input: Tensor::from_primitive(TensorPrimitive::Float(output.receptance_input)),
        weight_decay_input: Tensor::from_primitive(TensorPrimitive::Float(
            output.weight_decay_input,
        )),
        key_input: Tensor::from_primitive(TensorPrimitive::Float(output.key_input)),
        value_input: Tensor::from_primitive(TensorPrimitive::Float(output.value_input)),
        learning_rate_input: Tensor::from_primitive(TensorPrimitive::Float(
            output.learning_rate_input,
        )),
    }
}

/// Computes five addcmul outputs with regular tensor operations as the semantic reference.
///
/// Each output has the same shape as `base`.
/// The clones below are Burn tensor handle clones needed to express five ownership-consuming
/// formulas; they are not intended as eager data copies. The redundancy is in the reference
/// expression graph: each branch separately reads `base`, `diff`, and its scale, then performs a
/// generic multiply and add. The reference does not use slicing or concatenation, so the custom
/// kernel targets repeated arithmetic and memory traffic rather than slice/concat overhead.
pub fn addcmul5_reference<B: Backend>(
    inputs: Addcmul5ForwardInputs<B>,
) -> Addcmul5ForwardOutput<B> {
    Addcmul5ForwardOutput {
        receptance_input: inputs.base.clone() + inputs.diff.clone() * inputs.receptance_scale,
        weight_decay_input: inputs.base.clone() + inputs.diff.clone() * inputs.weight_decay_scale,
        key_input: inputs.base.clone() + inputs.diff.clone() * inputs.key_scale,
        value_input: inputs.base.clone() + inputs.diff.clone() * inputs.value_scale,
        learning_rate_input: inputs.base + inputs.diff * inputs.learning_rate_scale,
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Distribution, Tensor, Tolerance};

    use crate::{
        kernels::addcmul::{
            addcmul_custom,
            addcmul_reference,
            addcmul5_custom,
            addcmul5_reference,
            io::{Addcmul5ForwardInputs, AddcmulForwardInputs},
        },
        test_utils::backend::{TestAutodiffBackend, TestBackend},
    };

    /// Please name this function "forward" when you reuse this template.
    #[test]
    fn forward() {
        let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();

        let base = Tensor::<TestBackend, 3>::random([2, 8, 32], Distribution::Default, &device);
        let diff = Tensor::random([2, 8, 32], Distribution::Default, &device);
        let scale = Tensor::random([1, 1, 32], Distribution::Default, &device);

        let inputs = AddcmulForwardInputs {
            base: base.clone(),
            diff: diff.clone(),
            scale,
        };

        let reference = addcmul_reference(inputs.clone())
            .into_data()
            .convert::<f32>();
        let custom = addcmul_custom(inputs).into_data().convert::<f32>();

        reference.assert_approx_eq::<f32>(&custom, Tolerance::default());

        let inputs = Addcmul5ForwardInputs {
            base,
            diff,
            receptance_scale: Tensor::random([1, 1, 32], Distribution::Default, &device),
            weight_decay_scale: Tensor::random([1, 1, 32], Distribution::Default, &device),
            key_scale: Tensor::random([1, 1, 32], Distribution::Default, &device),
            value_scale: Tensor::random([1, 1, 32], Distribution::Default, &device),
            learning_rate_scale: Tensor::random([1, 1, 32], Distribution::Default, &device),
        };

        let reference = addcmul5_reference(inputs.clone());
        let custom = addcmul5_custom(inputs);

        reference
            .receptance_input
            .into_data()
            .convert::<f32>()
            .assert_approx_eq::<f32>(
                &custom.receptance_input.into_data().convert::<f32>(),
                Tolerance::default(),
            );
        reference
            .weight_decay_input
            .into_data()
            .convert::<f32>()
            .assert_approx_eq::<f32>(
                &custom.weight_decay_input.into_data().convert::<f32>(),
                Tolerance::default(),
            );
        reference
            .key_input
            .into_data()
            .convert::<f32>()
            .assert_approx_eq::<f32>(
                &custom.key_input.into_data().convert::<f32>(),
                Tolerance::default(),
            );
        reference
            .value_input
            .into_data()
            .convert::<f32>()
            .assert_approx_eq::<f32>(
                &custom.value_input.into_data().convert::<f32>(),
                Tolerance::default(),
            );
        reference
            .learning_rate_input
            .into_data()
            .convert::<f32>()
            .assert_approx_eq::<f32>(
                &custom.learning_rate_input.into_data().convert::<f32>(),
                Tolerance::default(),
            );

        println!("Both reference and the custom fused addcmul kernels have the same output");
    }

    /// Please name this function "backward" when you reuse this template.
    #[test]
    fn backward() {
        let device: <TestAutodiffBackend as burn::tensor::backend::Backend>::Device =
            Default::default();

        let base =
            Tensor::<TestAutodiffBackend, 3>::random([2, 8, 32], Distribution::Default, &device)
                .require_grad();
        let diff =
            Tensor::<TestAutodiffBackend, 3>::random([2, 8, 32], Distribution::Default, &device)
                .require_grad();
        let scale =
            Tensor::<TestAutodiffBackend, 3>::random([1, 1, 32], Distribution::Default, &device)
                .require_grad();

        let reference = addcmul_reference(AddcmulForwardInputs {
            base: base.clone(),
            diff: diff.clone(),
            scale: scale.clone(),
        });

        let mut gradients = reference.backward();

        let base_grad_ref = base.grad_remove(&mut gradients).unwrap();
        let diff_grad_ref = diff.grad_remove(&mut gradients).unwrap();
        let scale_grad_ref = scale.grad_remove(&mut gradients).unwrap();

        let base_custom = base.detach().require_grad();
        let diff_custom = diff.detach().require_grad();
        let scale_custom = scale.detach().require_grad();

        let custom = addcmul_custom(AddcmulForwardInputs {
            base: base_custom.clone(),
            diff: diff_custom.clone(),
            scale: scale_custom.clone(),
        });

        let mut gradients = custom.backward();

        let base_grad_custom = base_custom.grad_remove(&mut gradients).unwrap();
        let diff_grad_custom = diff_custom.grad_remove(&mut gradients).unwrap();
        let scale_grad_custom = scale_custom.grad_remove(&mut gradients).unwrap();

        base_grad_ref
            .into_data()
            .convert::<f32>()
            .assert_approx_eq::<f32>(
                &base_grad_custom.into_data().convert::<f32>(),
                Tolerance::default(),
            );
        diff_grad_ref
            .into_data()
            .convert::<f32>()
            .assert_approx_eq::<f32>(
                &diff_grad_custom.into_data().convert::<f32>(),
                Tolerance::default(),
            );
        scale_grad_ref
            .into_data()
            .convert::<f32>()
            .assert_approx_eq::<f32>(
                &scale_grad_custom.into_data().convert::<f32>(),
                Tolerance::default(),
            );

        println!("Both reference and the custom fused addcmul kernel have the same gradients");

        let base =
            Tensor::<TestAutodiffBackend, 3>::random([2, 8, 32], Distribution::Default, &device)
                .require_grad();
        let diff =
            Tensor::<TestAutodiffBackend, 3>::random([2, 8, 32], Distribution::Default, &device)
                .require_grad();
        let receptance_scale =
            Tensor::<TestAutodiffBackend, 3>::random([1, 1, 32], Distribution::Default, &device)
                .require_grad();
        let weight_decay_scale =
            Tensor::<TestAutodiffBackend, 3>::random([1, 1, 32], Distribution::Default, &device)
                .require_grad();
        let key_scale =
            Tensor::<TestAutodiffBackend, 3>::random([1, 1, 32], Distribution::Default, &device)
                .require_grad();
        let value_scale =
            Tensor::<TestAutodiffBackend, 3>::random([1, 1, 32], Distribution::Default, &device)
                .require_grad();
        let learning_rate_scale =
            Tensor::<TestAutodiffBackend, 3>::random([1, 1, 32], Distribution::Default, &device)
                .require_grad();

        let reference = addcmul5_reference(Addcmul5ForwardInputs {
            base: base.clone(),
            diff: diff.clone(),
            receptance_scale: receptance_scale.clone(),
            weight_decay_scale: weight_decay_scale.clone(),
            key_scale: key_scale.clone(),
            value_scale: value_scale.clone(),
            learning_rate_scale: learning_rate_scale.clone(),
        });
        let reference = reference.receptance_input
            + reference.weight_decay_input
            + reference.key_input
            + reference.value_input
            + reference.learning_rate_input;

        let mut gradients = reference.backward();

        let base_grad_ref = base.grad_remove(&mut gradients).unwrap();
        let diff_grad_ref = diff.grad_remove(&mut gradients).unwrap();
        let receptance_scale_grad_ref = receptance_scale.grad_remove(&mut gradients).unwrap();
        let weight_decay_scale_grad_ref = weight_decay_scale.grad_remove(&mut gradients).unwrap();
        let key_scale_grad_ref = key_scale.grad_remove(&mut gradients).unwrap();
        let value_scale_grad_ref = value_scale.grad_remove(&mut gradients).unwrap();
        let learning_rate_scale_grad_ref = learning_rate_scale.grad_remove(&mut gradients).unwrap();

        let base_custom = base.detach().require_grad();
        let diff_custom = diff.detach().require_grad();
        let receptance_scale_custom = receptance_scale.detach().require_grad();
        let weight_decay_scale_custom = weight_decay_scale.detach().require_grad();
        let key_scale_custom = key_scale.detach().require_grad();
        let value_scale_custom = value_scale.detach().require_grad();
        let learning_rate_scale_custom = learning_rate_scale.detach().require_grad();

        let custom = addcmul5_custom(Addcmul5ForwardInputs {
            base: base_custom.clone(),
            diff: diff_custom.clone(),
            receptance_scale: receptance_scale_custom.clone(),
            weight_decay_scale: weight_decay_scale_custom.clone(),
            key_scale: key_scale_custom.clone(),
            value_scale: value_scale_custom.clone(),
            learning_rate_scale: learning_rate_scale_custom.clone(),
        });
        let custom = custom.receptance_input
            + custom.weight_decay_input
            + custom.key_input
            + custom.value_input
            + custom.learning_rate_input;

        let mut gradients = custom.backward();

        let base_grad_custom = base_custom.grad_remove(&mut gradients).unwrap();
        let diff_grad_custom = diff_custom.grad_remove(&mut gradients).unwrap();
        let receptance_scale_grad_custom =
            receptance_scale_custom.grad_remove(&mut gradients).unwrap();
        let weight_decay_scale_grad_custom = weight_decay_scale_custom
            .grad_remove(&mut gradients)
            .unwrap();
        let key_scale_grad_custom = key_scale_custom.grad_remove(&mut gradients).unwrap();
        let value_scale_grad_custom = value_scale_custom.grad_remove(&mut gradients).unwrap();
        let learning_rate_scale_grad_custom = learning_rate_scale_custom
            .grad_remove(&mut gradients)
            .unwrap();

        base_grad_ref
            .into_data()
            .convert::<f32>()
            .assert_approx_eq::<f32>(
                &base_grad_custom.into_data().convert::<f32>(),
                Tolerance::default(),
            );
        diff_grad_ref
            .into_data()
            .convert::<f32>()
            .assert_approx_eq::<f32>(
                &diff_grad_custom.into_data().convert::<f32>(),
                Tolerance::default(),
            );
        receptance_scale_grad_ref
            .into_data()
            .convert::<f32>()
            .assert_approx_eq::<f32>(
                &receptance_scale_grad_custom.into_data().convert::<f32>(),
                Tolerance::default(),
            );
        weight_decay_scale_grad_ref
            .into_data()
            .convert::<f32>()
            .assert_approx_eq::<f32>(
                &weight_decay_scale_grad_custom.into_data().convert::<f32>(),
                Tolerance::default(),
            );
        key_scale_grad_ref
            .into_data()
            .convert::<f32>()
            .assert_approx_eq::<f32>(
                &key_scale_grad_custom.into_data().convert::<f32>(),
                Tolerance::default(),
            );
        value_scale_grad_ref
            .into_data()
            .convert::<f32>()
            .assert_approx_eq::<f32>(
                &value_scale_grad_custom.into_data().convert::<f32>(),
                Tolerance::default(),
            );
        learning_rate_scale_grad_ref
            .into_data()
            .convert::<f32>()
            .assert_approx_eq::<f32>(
                &learning_rate_scale_grad_custom.into_data().convert::<f32>(),
                Tolerance::default(),
            );

        println!("Both reference and the custom fused addcmul5 kernel have the same gradients");
    }
}
