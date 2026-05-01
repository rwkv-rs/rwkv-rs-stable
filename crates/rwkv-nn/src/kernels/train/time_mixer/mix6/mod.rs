mod backward;
mod forward;
/// Input and output containers for the RWKV7 pretrain mix6 kernel.
pub mod io;
mod kernel;

use burn::tensor::{Tensor, TensorPrimitive};
use burn_cubecl::{
    CubeBackend,
    CubeElement,
    CubeRuntime,
    FloatElement,
    IntElement,
    element::BoolElement,
};

use crate::kernels::train::time_mixer::mix6::io::{
    Mix6ForwardInputs,
    Mix6ForwardOutput,
    Mix6ForwardPrimitiveInputs,
    Mix6ForwardPrimitiveOutput,
};

/// Backend primitive capability for the RWKV7 pretrain mix6 operation.
pub trait Mix6Backend: burn::tensor::backend::Backend {
    /// Runs the six-output time-mix primitive.
    fn fused_mix6(inputs: Mix6ForwardPrimitiveInputs<Self>) -> Mix6ForwardPrimitiveOutput<Self>;
}

/// Autodiff backend marker for the mix6 kernel.
pub trait AutodiffBackend: Mix6Backend + burn::tensor::backend::AutodiffBackend {}

impl<B> AutodiffBackend for B where B: Mix6Backend + burn::tensor::backend::AutodiffBackend {}

impl<R, F, I, BT> Mix6Backend for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
{
    fn fused_mix6(inputs: Mix6ForwardPrimitiveInputs<Self>) -> Mix6ForwardPrimitiveOutput<Self> {
        assert!(
            inputs.embedded_context.is_contiguous(),
            "embedded_context must be contiguous"
        );
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
        assert!(
            inputs.gate_scale.is_contiguous(),
            "gate_scale must be contiguous"
        );

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

#[cfg(test)]
mod tests {
    use burn::tensor::{Distribution, ElementConversion, Tensor};

    use crate::{
        kernels::train::time_mixer::mix6::{
            io::{Mix6ForwardInputs, Mix6ForwardOutput},
            mix6_custom,
            mix6_reference,
        },
        test_utils::backend::{TestAutodiffBackend, TestAutodiffDevice, TestBackend, TestDevice},
    };

    #[test]
    fn forward() {
        let device: TestDevice = Default::default();

        for shape in [[2, 8, 32], [1, 3, 17]] {
            let inputs = random_inputs::<TestBackend>(shape, &device);

            let reference = mix6_reference(inputs.clone().into_inputs());
            let custom = mix6_custom(inputs.into_inputs());

            assert_output_close(reference, custom);
        }
    }

    #[test]
    fn backward() {
        let device: TestAutodiffDevice = Default::default();

        for shape in [[2, 8, 32], [1, 3, 17]] {
            for branch_index in 0..6 {
                let inputs = random_inputs::<TestAutodiffBackend>(shape, &device);
                let reference_inputs = inputs.clone().require_grad();
                let reference = branch_output(
                    mix6_reference(reference_inputs.clone().into_inputs()),
                    branch_index,
                );
                let mut gradients = reference.backward();
                let mut reference_grads = reference_inputs.remove_optional_grads(&mut gradients);

                let custom_inputs = inputs.require_grad();
                let custom = branch_output(
                    mix6_custom(custom_inputs.clone().into_inputs()),
                    branch_index,
                );
                let mut gradients = custom.backward();
                let mut custom_grads = custom_inputs.remove_optional_grads(&mut gradients);

                assert_tensor_close(
                    reference_grads[0].take().unwrap(),
                    custom_grads[0].take().unwrap(),
                );
                assert_tensor_close(
                    reference_grads[branch_index + 1].take().unwrap(),
                    custom_grads[branch_index + 1].take().unwrap(),
                );

                for scale_grad_index in 1..7 {
                    if scale_grad_index != branch_index + 1 {
                        assert_optional_zero(reference_grads[scale_grad_index].take());
                        assert_optional_zero(custom_grads[scale_grad_index].take());
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic]
    fn forward_panics_on_wrong_scale_shape() {
        let device: TestDevice = Default::default();
        let mut inputs = random_inputs::<TestBackend>([2, 4, 8], &device);
        inputs.gate_scale =
            Tensor::<TestBackend, 3>::random([1, 1, 7], Distribution::Default, &device);

        let _ = mix6_custom(inputs.into_inputs());
    }

    #[derive(Clone)]
    struct TestInputs<B: super::Mix6Backend> {
        embedded_context: Tensor<B, 3>,
        receptance_scale: Tensor<B, 3>,
        weight_decay_scale: Tensor<B, 3>,
        key_scale: Tensor<B, 3>,
        value_scale: Tensor<B, 3>,
        learning_rate_scale: Tensor<B, 3>,
        gate_scale: Tensor<B, 3>,
    }

    impl<B: super::Mix6Backend> TestInputs<B> {
        fn into_inputs(self) -> Mix6ForwardInputs<B> {
            Mix6ForwardInputs {
                embedded_context: self.embedded_context,
                receptance_scale: self.receptance_scale,
                weight_decay_scale: self.weight_decay_scale,
                key_scale: self.key_scale,
                value_scale: self.value_scale,
                learning_rate_scale: self.learning_rate_scale,
                gate_scale: self.gate_scale,
            }
        }
    }

    impl<B: super::AutodiffBackend> TestInputs<B> {
        fn require_grad(self) -> Self {
            Self {
                embedded_context: self.embedded_context.require_grad(),
                receptance_scale: self.receptance_scale.require_grad(),
                weight_decay_scale: self.weight_decay_scale.require_grad(),
                key_scale: self.key_scale.require_grad(),
                value_scale: self.value_scale.require_grad(),
                learning_rate_scale: self.learning_rate_scale.require_grad(),
                gate_scale: self.gate_scale.require_grad(),
            }
        }

        fn remove_optional_grads(
            self,
            gradients: &mut B::Gradients,
        ) -> Vec<Option<Tensor<B::InnerBackend, 3>>> {
            vec![
                self.embedded_context.grad_remove(gradients),
                self.receptance_scale.grad_remove(gradients),
                self.weight_decay_scale.grad_remove(gradients),
                self.key_scale.grad_remove(gradients),
                self.value_scale.grad_remove(gradients),
                self.learning_rate_scale.grad_remove(gradients),
                self.gate_scale.grad_remove(gradients),
            ]
        }
    }

    fn random_inputs<B: super::Mix6Backend>(
        shape: [usize; 3],
        device: &B::Device,
    ) -> TestInputs<B> {
        TestInputs {
            embedded_context: Tensor::<B, 3>::random(shape, Distribution::Default, device),
            receptance_scale: Tensor::<B, 3>::random(
                [1, 1, shape[2]],
                Distribution::Default,
                device,
            ),
            weight_decay_scale: Tensor::<B, 3>::random(
                [1, 1, shape[2]],
                Distribution::Default,
                device,
            ),
            key_scale: Tensor::<B, 3>::random([1, 1, shape[2]], Distribution::Default, device),
            value_scale: Tensor::<B, 3>::random([1, 1, shape[2]], Distribution::Default, device),
            learning_rate_scale: Tensor::<B, 3>::random(
                [1, 1, shape[2]],
                Distribution::Default,
                device,
            ),
            gate_scale: Tensor::<B, 3>::random([1, 1, shape[2]], Distribution::Default, device),
        }
    }

    fn branch_output<B: super::Mix6Backend>(
        output: Mix6ForwardOutput<B>,
        branch_index: usize,
    ) -> Tensor<B, 3> {
        match branch_index {
            0 => output.receptance_input,
            1 => output.weight_decay_input,
            2 => output.key_input,
            3 => output.value_input,
            4 => output.learning_rate_input,
            5 => output.gate_input,
            _ => panic!("branch index must be in 0..6"),
        }
    }

    fn assert_output_close<B: super::Mix6Backend>(
        reference: Mix6ForwardOutput<B>,
        custom: Mix6ForwardOutput<B>,
    ) {
        assert_tensor_close(reference.receptance_input, custom.receptance_input);
        assert_tensor_close(reference.weight_decay_input, custom.weight_decay_input);
        assert_tensor_close(reference.key_input, custom.key_input);
        assert_tensor_close(reference.value_input, custom.value_input);
        assert_tensor_close(reference.learning_rate_input, custom.learning_rate_input);
        assert_tensor_close(reference.gate_input, custom.gate_input);
    }

    fn assert_tensor_close<B: super::Mix6Backend, const D: usize>(
        reference: Tensor<B, D>,
        custom: Tensor<B, D>,
    ) {
        assert_eq!(reference.dims(), custom.dims());
        if reference.shape().num_elements() == 0 {
            return;
        }

        let max_diff = (reference - custom).abs().max().into_scalar().elem::<f32>();
        assert!(max_diff <= 1.0e-4, "max diff {max_diff}");
    }

    fn assert_optional_zero<B: super::Mix6Backend, const D: usize>(tensor: Option<Tensor<B, D>>) {
        let Some(tensor) = tensor else {
            return;
        };
        if tensor.shape().num_elements() == 0 {
            return;
        }

        let max_abs = tensor.abs().max().into_scalar().elem::<f32>();
        assert!(max_abs <= 1.0e-4, "max abs {max_abs}");
    }
}
