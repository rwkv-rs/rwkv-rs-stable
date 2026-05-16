mod backward;
mod forward;
/// Public input wrappers for the matmul-add-ReLU template kernel.
pub mod io;
mod kernel;

use burn::tensor::{Tensor, TensorPrimitive, activation, ops::FloatTensor};

use self::io::{MatmulAddReluInputs, MatmulAddReluPrimitiveInputs};

/// We create our own Backend trait that extends the Burn backend trait.
pub trait MatmulAddReluBackend: burn::tensor::backend::Backend {
    /// Please name this function "fused_{kernel_name}" when you reuse this template.
    fn fused_matmul_add_relu(inputs: MatmulAddReluPrimitiveInputs<Self>) -> FloatTensor<Self>;
}

/// We create our own AutodiffBackend trait that extends the Burn autodiff backend trait.
pub trait AutodiffBackend: MatmulAddReluBackend + burn::tensor::backend::AutodiffBackend {}

/// We define our custom implementation using the added function on our custom backend.
/// Please name this function "{kernel_name}_custom" when you reuse this template.
pub fn matmul_add_relu_custom<B: MatmulAddReluBackend>(
    inputs: MatmulAddReluInputs<B>,
) -> Tensor<B, 3> {
    inputs.check().unwrap();

    let output = B::fused_matmul_add_relu(inputs.to_primitive());

    Tensor::from_primitive(TensorPrimitive::Float(output))
}

/// We define a reference implementation using basic tensor operations.
/// Please name this function "{kernel_name}_reference" when you reuse this template.
pub fn matmul_add_relu_reference<B: MatmulAddReluBackend>(
    inputs: MatmulAddReluInputs<B>,
) -> Tensor<B, 3> {
    let x = inputs.lhs.matmul(inputs.rhs) + inputs.bias;

    activation::relu(x)
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Distribution, Tensor};

    use crate::{
        kernels::template::matmul_add_relu::{
            io::MatmulAddReluInputs,
            matmul_add_relu_custom,
            matmul_add_relu_reference,
        },
        test_utils::{
            backend::{TestAutodiffBackend, TestAutodiffDevice, TestBackend, TestDevice},
            numeric::{NumericTolerance, assert_tensor_close},
        },
    };

    const TOLERANCE: NumericTolerance = NumericTolerance::new(3.0e-3, 5.0e-4, 0.999999);

    /// Please name this function "forward" when you reuse this template.
    #[test]
    fn forward() {
        let device: TestDevice = Default::default();

        let lhs = Tensor::<TestBackend, 3>::random([1, 32, 32], Distribution::Default, &device);
        let rhs = Tensor::random([32, 32, 32], Distribution::Default, &device);
        let bias = Tensor::random([32, 32, 32], Distribution::Default, &device);

        let inputs = MatmulAddReluInputs { lhs, rhs, bias };

        let reference = matmul_add_relu_reference(inputs.clone());
        let custom = matmul_add_relu_custom(inputs);

        assert_tensor_close(
            "kernels/template/matmul_add_relu/forward",
            custom,
            reference,
            TOLERANCE,
        );

        println!("Both reference and the custom fused kernel have the same output");
    }

    /// Please name this function "backward" when you reuse this template.
    #[test]
    fn backward() {
        let device: TestAutodiffDevice = Default::default();

        let lhs =
            Tensor::<TestAutodiffBackend, 3>::random([1, 32, 32], Distribution::Default, &device)
                .require_grad();
        let rhs =
            Tensor::<TestAutodiffBackend, 3>::random([32, 32, 32], Distribution::Default, &device)
                .require_grad();
        let bias =
            Tensor::<TestAutodiffBackend, 3>::random([32, 32, 32], Distribution::Default, &device)
                .require_grad();

        let inputs = MatmulAddReluInputs {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            bias: bias.clone(),
        };

        let reference = matmul_add_relu_reference(inputs.clone());

        let mut gradients = reference.backward();

        let lhs_grad_ref = lhs.grad_remove(&mut gradients).unwrap();
        let rhs_grad_ref = rhs.grad_remove(&mut gradients).unwrap();
        let bias_grad_ref = bias.grad_remove(&mut gradients).unwrap();

        let lhs_custom = lhs.detach().require_grad();
        let rhs_custom = rhs.detach().require_grad();
        let bias_custom = bias.detach().require_grad();

        let custom = matmul_add_relu_custom(MatmulAddReluInputs {
            lhs: lhs_custom.clone(),
            rhs: rhs_custom.clone(),
            bias: bias_custom.clone(),
        });

        let mut gradients = custom.backward();

        let lhs_grad_custom = lhs_custom.grad_remove(&mut gradients).unwrap();
        let rhs_grad_custom = rhs_custom.grad_remove(&mut gradients).unwrap();
        let bias_grad_custom = bias_custom.grad_remove(&mut gradients).unwrap();

        assert_tensor_close(
            "kernels/template/matmul_add_relu/backward/lhs",
            lhs_grad_custom,
            lhs_grad_ref,
            TOLERANCE,
        );

        println!("Both reference and the custom fused kernel have the same lhs gradient");

        assert_tensor_close(
            "kernels/template/matmul_add_relu/backward/rhs",
            rhs_grad_custom,
            rhs_grad_ref,
            TOLERANCE,
        );

        println!("Both reference and the custom fused kernel have the same rhs gradient");

        assert_tensor_close(
            "kernels/template/matmul_add_relu/backward/bias",
            bias_grad_custom,
            bias_grad_ref,
            TOLERANCE,
        );

        println!("Both reference and the custom fused kernel have the same bias gradient");
    }
}
