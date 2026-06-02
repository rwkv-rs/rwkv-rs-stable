mod forward;
/// Input containers for the fused residual-add kernel.
pub mod io;
mod kernel;

use burn::{
    backend::autodiff::{Autodiff, checkpoint::strategy::CheckpointStrategy},
    prelude::Backend,
    tensor::{Tensor, TensorPrimitive, ops::FloatTensor},
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
    residual_add::io::ResidualAddPrimitiveInputs,
};

/// Backend primitive capability for fused residual addition.
pub trait ResidualAddBackend: burn::tensor::backend::Backend {
    /// Runs the fused residual-add primitive.
    fn fused_residual_add(inputs: ResidualAddPrimitiveInputs<Self>) -> FloatTensor<Self>;
}

impl<R, F, I, BT> ResidualAddBackend for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
{
    fn fused_residual_add(inputs: ResidualAddPrimitiveInputs<Self>) -> FloatTensor<Self> {
        assert_linear_readable("lhs", &inputs.lhs);
        assert_linear_readable("rhs", &inputs.rhs);

        forward::fused_residual_add::<R, F, I, BT>(inputs)
    }
}

impl<B, C> ResidualAddBackend for Autodiff<B, C>
where
    B: Backend,
    C: CheckpointStrategy,
{
    fn fused_residual_add(inputs: ResidualAddPrimitiveInputs<Self>) -> FloatTensor<Self> {
        let lhs = Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(inputs.lhs));
        let rhs = Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(inputs.rhs));

        (lhs + rhs).into_primitive().tensor()
    }
}

/// Adds two same-shaped 3D activation tensors.
pub fn residual_add<B: ResidualAddBackend>(lhs: Tensor<B, 3>, rhs: Tensor<B, 3>) -> Tensor<B, 3> {
    let output = B::fused_residual_add(ResidualAddPrimitiveInputs {
        lhs: lhs.into_primitive().tensor(),
        rhs: rhs.into_primitive().tensor(),
    });

    Tensor::from_primitive(TensorPrimitive::Float(output))
}
