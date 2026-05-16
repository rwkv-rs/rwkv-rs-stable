mod forward;
/// Input containers for the fused layer-normalization kernel.
pub mod io;
mod kernel;

use burn::tensor::{ops::FloatTensor, Tensor, TensorPrimitive};
use burn_cubecl::{
    element::BoolElement,
    CubeBackend,
    CubeElement,
    CubeRuntime,
    FloatElement,
    IntElement,
};

use crate::kernels::train::{
    layer_norm::io::LayerNormPrimitiveInputs,
    layout::assert_linear_readable,
};

/// Backend primitive capability for fused layer normalization.
pub trait LayerNormBackend: burn::tensor::backend::Backend {
    /// Runs the fused layer-normalization primitive.
    fn fused_layer_norm(inputs: LayerNormPrimitiveInputs<Self>) -> FloatTensor<Self>;
}

impl<R, F, I, BT> LayerNormBackend for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
{
    fn fused_layer_norm(inputs: LayerNormPrimitiveInputs<Self>) -> FloatTensor<Self> {
        assert_linear_readable("input", &inputs.input);
        assert_linear_readable("gamma", &inputs.gamma);
        assert_linear_readable("beta", &inputs.beta);

        forward::fused_layer_norm::<R, F, I, BT>(inputs)
    }
}

/// Runs fused layer normalization over the last dimension of a 3D RWKV activation tensor.
pub fn layer_norm<B: LayerNormBackend>(
    input: Tensor<B, 3>,
    gamma: Tensor<B, 1>,
    beta: Tensor<B, 1>,
    epsilon: f64,
) -> Tensor<B, 3> {
    let output = B::fused_layer_norm(LayerNormPrimitiveInputs {
        input: input.into_primitive().tensor(),
        gamma: gamma.into_primitive().tensor(),
        beta: beta.into_primitive().tensor(),
        epsilon,
    });

    Tensor::from_primitive(TensorPrimitive::Float(output))
}
