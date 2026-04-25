use burn::{
    cubecl::{CubeCount, CubeDim},
    tensor::{ops::FloatTensor, Shape},
};
use burn_cubecl::{
    element::BoolElement,
    tensor::CubeTensor,
    CubeBackend,
    CubeRuntime,
    FloatElement,
    IntElement,
};

use crate::kernels::template::{
    io::MatmulAddReluPrimitiveInputs,
    kernel::fused_matmul_add_relu_kernel,
    TemplateBackend,
};

/// Implement our custom backend trait for the generic `CubeBackend`.
impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> TemplateBackend
    for CubeBackend<R, F, I, BT>
{
    fn fused_matmul_add_relu(inputs: MatmulAddReluPrimitiveInputs<Self>) -> FloatTensor<Self> {
        // Define cube dim, hardcoded for simplicity.
        let cube_dim = CubeDim { x: 16, y: 16, z: 1 };

        // Get the matmul relevant shapes.
        let ndims = inputs.lhs.meta.num_dims();
        let num_rows = inputs.lhs.meta.shape()[ndims - 2];
        let num_cols = inputs.rhs.meta.shape()[ndims - 1];

        // Compute shape of output, while tracking number of batches.
        let mut num_batches = 1;
        let mut shape_out = vec![0; ndims];
        for i in 0..ndims - 2 {
            shape_out[i] = usize::max(inputs.lhs.meta.shape()[i], inputs.rhs.meta.shape()[i]);
            num_batches *= shape_out[i];
        }
        shape_out[ndims - 2] = num_rows;
        shape_out[ndims - 1] = num_cols;
        let shape_out = Shape::from(shape_out);

        // Create a buffer for the output tensor.
        let buffer = inputs
            .lhs
            .client
            .empty(shape_out.num_elements() * core::mem::size_of::<F>());

        // Create the output tensor primitive.
        let output = CubeTensor::new_contiguous(
            inputs.lhs.client.clone(),
            inputs.lhs.device.clone(),
            shape_out,
            buffer,
            F::dtype(),
        );

        // Declare the wgsl workgroup with the number of cubes in x, y and z.
        let cubes_needed_in_x = f32::ceil(num_rows as f32 / cube_dim.x as f32) as u32;
        let cubes_needed_in_y = f32::ceil(num_cols as f32 / cube_dim.y as f32) as u32;
        let cube_count =
            CubeCount::Static(cubes_needed_in_x, cubes_needed_in_y, num_batches as u32);

        // Execute lazily the kernel with the launch information and the given buffers. For
        // simplicity, no vectorization is performed
        fused_matmul_add_relu_kernel::launch::<F, R>(
            &output.client,
            cube_count,
            cube_dim,
            inputs.lhs.into_tensor_arg(),
            inputs.rhs.into_tensor_arg(),
            inputs.bias.into_tensor_arg(),
            output.clone().into_tensor_arg(),
        );

        // Return the output tensor.
        output
    }
}

#[cfg(feature = "fusion")]
mod fusion_impl {
    use burn::tensor::ops::{ActivationOps, FloatTensorOps};
    use burn_fusion::{Fusion, FusionBackend};

    use super::*;

    impl<B: FusionBackend + TemplateBackend> TemplateBackend for Fusion<B> {
        fn fused_matmul_add_relu(inputs: MatmulAddReluPrimitiveInputs<Self>) -> FloatTensor<Self> {
            Self::relu(Self::float_add(
                Self::float_matmul(inputs.lhs, inputs.rhs),
                inputs.bias,
            ))
        }
    }
}
