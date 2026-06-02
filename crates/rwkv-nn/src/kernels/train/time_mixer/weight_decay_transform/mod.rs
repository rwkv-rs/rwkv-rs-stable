mod forward;
/// Input containers for the weight-decay transform kernel.
pub mod io;
mod kernel;

use burn::{
    backend::autodiff::{Autodiff, checkpoint::strategy::CheckpointStrategy},
    prelude::Backend,
    tensor::{Tensor, TensorPrimitive, activation::softplus, ops::FloatTensor},
};
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
    time_mixer::weight_decay_transform::io::{
        WeightDecayTransformForwardInputs,
        WeightDecayTransformForwardPrimitiveInputs,
    },
};

/// Backend primitive capability for the fused weight-decay transform.
pub trait WeightDecayTransformBackend: burn::tensor::backend::Backend {
    /// Runs `-softplus(-(weight_decay_base + weight_decay_input)) - 0.5` as one fused primitive.
    fn fused_weight_decay_transform(
        inputs: WeightDecayTransformForwardPrimitiveInputs<Self>,
    ) -> FloatTensor<Self>;
}

impl<R, F, I, BT> WeightDecayTransformBackend for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
{
    fn fused_weight_decay_transform(
        inputs: WeightDecayTransformForwardPrimitiveInputs<Self>,
    ) -> FloatTensor<Self> {
        assert_linear_readable("weight_decay_base", &inputs.weight_decay_base);
        assert_linear_readable("weight_decay_input", &inputs.weight_decay_input);

        forward::fused_weight_decay_transform::<R, F, I, BT>(inputs)
    }
}

impl<B, C> WeightDecayTransformBackend for Autodiff<B, C>
where
    B: Backend,
    C: CheckpointStrategy,
{
    fn fused_weight_decay_transform(
        inputs: WeightDecayTransformForwardPrimitiveInputs<Self>,
    ) -> FloatTensor<Self> {
        weight_decay_transform_reference(
            Tensor::<Self, 1>::from_primitive(TensorPrimitive::Float(inputs.weight_decay_base)),
            Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(inputs.weight_decay_input)),
        )
        .into_primitive()
        .tensor()
    }
}

/// Runs the fused RWKV7 weight-decay transform after validating the public input contract.
pub fn weight_decay_transform_custom<B: WeightDecayTransformBackend>(
    inputs: WeightDecayTransformForwardInputs<B>,
) -> Tensor<B, 3> {
    inputs.check().unwrap();
    let output = B::fused_weight_decay_transform(inputs.to_primitive());

    Tensor::from_primitive(TensorPrimitive::Float(output))
}

/// Computes the weight-decay transform with regular Burn tensor operations.
pub fn weight_decay_transform_reference<B: Backend>(
    weight_decay_base: Tensor<B, 1>,
    weight_decay_input: Tensor<B, 3>,
) -> Tensor<B, 3> {
    -softplus(
        -(weight_decay_input
            + weight_decay_base
                .unsqueeze_dim::<2>(0)
                .unsqueeze_dim::<3>(0)),
        1.0,
    ) - 0.5
}

/// Convenience wrapper for the differentiable weight-decay transform path.
pub fn weight_decay_transform<B: WeightDecayTransformBackend>(
    weight_decay_base: Tensor<B, 1>,
    weight_decay_input: Tensor<B, 3>,
) -> Tensor<B, 3> {
    weight_decay_transform_custom(WeightDecayTransformForwardInputs {
        weight_decay_base,
        weight_decay_input,
    })
}
