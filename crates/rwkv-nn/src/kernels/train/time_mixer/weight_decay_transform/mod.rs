mod forward;
/// Input containers for the weight-decay transform kernel.
pub mod io;
mod kernel;

use burn::tensor::{Tensor, TensorPrimitive, activation::softplus, ops::FloatTensor};
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

/// Runs the fused RWKV7 weight-decay transform after validating the public input contract.
pub fn weight_decay_transform_custom<B: WeightDecayTransformBackend>(
    inputs: WeightDecayTransformForwardInputs<B>,
) -> Tensor<B, 3> {
    inputs.check().unwrap();
    let output = B::fused_weight_decay_transform(inputs.to_primitive());

    Tensor::from_primitive(TensorPrimitive::Float(output))
}

/// Computes the weight-decay transform with regular Burn tensor operations.
pub fn weight_decay_transform_reference<B: WeightDecayTransformBackend>(
    inputs: WeightDecayTransformForwardInputs<B>,
) -> Tensor<B, 3> {
    -softplus(
        -(inputs.weight_decay_input
            + inputs
                .weight_decay_base
                .unsqueeze_dim::<2>(0)
                .unsqueeze_dim::<3>(0)),
        1.0,
    ) - 0.5
}

/// Convenience wrapper for the fused weight-decay transform path.
pub fn weight_decay_transform<B: WeightDecayTransformBackend>(
    weight_decay_base: Tensor<B, 1>,
    weight_decay_input: Tensor<B, 3>,
) -> Tensor<B, 3> {
    weight_decay_transform_custom(WeightDecayTransformForwardInputs {
        weight_decay_base,
        weight_decay_input,
    })
}
