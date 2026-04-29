mod backward;
mod forward;
/// Input and output containers for RWKV7 WKV kernels.
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

use crate::kernels::train::time_mixer::wkv7::io::{
    Wkv7PretrainForwardInputs,
    Wkv7PretrainForwardPrimitiveInputs,
    Wkv7StatepassForwardInputs,
    Wkv7StatepassForwardOutput,
    Wkv7StatepassForwardPrimitiveInputs,
    Wkv7StatepassForwardPrimitiveOutput,
    Wkv7StatetuneForwardInputs,
    Wkv7StatetuneForwardPrimitiveInputs,
};

/// Backend primitive capability for RWKV7 WKV training kernels.
pub trait Wkv7Backend: burn::tensor::backend::Backend {
    /// Runs the zero-initial-state RWKV7 WKV primitive.
    fn fused_wkv7_pretrain(inputs: Wkv7PretrainForwardPrimitiveInputs<Self>) -> FloatTensor<Self>;

    /// Runs the RWKV7 WKV primitive with a supplied initial state.
    fn fused_wkv7_statetune(inputs: Wkv7StatetuneForwardPrimitiveInputs<Self>)
    -> FloatTensor<Self>;

    /// Runs the RWKV7 WKV primitive with state carry-out.
    fn fused_wkv7_statepass(
        inputs: Wkv7StatepassForwardPrimitiveInputs<Self>,
    ) -> Wkv7StatepassForwardPrimitiveOutput<Self>;
}

/// Autodiff backend marker for RWKV7 WKV kernels.
pub trait AutodiffBackend: Wkv7Backend + burn::tensor::backend::AutodiffBackend {}

impl<B> AutodiffBackend for B where B: Wkv7Backend + burn::tensor::backend::AutodiffBackend {}

impl<R, F, I, BT> Wkv7Backend for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
{
    fn fused_wkv7_pretrain(inputs: Wkv7PretrainForwardPrimitiveInputs<Self>) -> FloatTensor<Self> {
        assert!(
            inputs.receptance.is_contiguous(),
            "receptance must be contiguous"
        );
        assert!(
            inputs.weight_decay.is_contiguous(),
            "weight_decay must be contiguous"
        );
        assert!(
            inputs.replacement_key.is_contiguous(),
            "replacement_key must be contiguous"
        );
        assert!(inputs.value.is_contiguous(), "value must be contiguous");
        assert!(
            inputs.removal_key_normalized.is_contiguous(),
            "removal_key_normalized must be contiguous"
        );
        assert!(
            inputs.replacement.is_contiguous(),
            "replacement must be contiguous"
        );

        forward::fused_wkv7_pretrain::<R, F, I, BT>(inputs)
    }

    fn fused_wkv7_statetune(
        inputs: Wkv7StatetuneForwardPrimitiveInputs<Self>,
    ) -> FloatTensor<Self> {
        assert!(
            inputs.initial_state.is_contiguous(),
            "initial_state must be contiguous"
        );

        forward::fused_wkv7_statetune::<R, F, I, BT>(inputs)
    }

    fn fused_wkv7_statepass(
        inputs: Wkv7StatepassForwardPrimitiveInputs<Self>,
    ) -> Wkv7StatepassForwardPrimitiveOutput<Self> {
        assert!(
            inputs.initial_state.is_contiguous(),
            "initial_state must be contiguous"
        );

        forward::fused_wkv7_statepass::<R, F, I, BT>(inputs)
    }
}

/// Runs the zero-initial-state RWKV7 WKV custom kernel.
pub fn wkv7_pretrain_custom<B: Wkv7Backend>(inputs: Wkv7PretrainForwardInputs<B>) -> Tensor<B, 4> {
    inputs.check().unwrap();
    Tensor::from_primitive(TensorPrimitive::Float(B::fused_wkv7_pretrain(
        inputs.to_primitive(),
    )))
}

/// Runs the supplied-initial-state RWKV7 WKV custom kernel.
pub fn wkv7_statetune_custom<B: Wkv7Backend>(
    inputs: Wkv7StatetuneForwardInputs<B>,
) -> Tensor<B, 4> {
    inputs.check().unwrap();
    Tensor::from_primitive(TensorPrimitive::Float(B::fused_wkv7_statetune(
        inputs.to_primitive(),
    )))
}

/// Runs the state-passing RWKV7 WKV custom kernel.
pub fn wkv7_statepass_custom<B: Wkv7Backend>(
    inputs: Wkv7StatepassForwardInputs<B>,
) -> Wkv7StatepassForwardOutput<B> {
    inputs.check().unwrap();
    let output = B::fused_wkv7_statepass(inputs.to_primitive());

    Wkv7StatepassForwardOutput {
        output: Tensor::from_primitive(TensorPrimitive::Float(output.output)),
        next_state: Tensor::from_primitive(TensorPrimitive::Float(output.next_state)),
    }
}

/// Convenience wrapper for the zero-state WKV path.
pub fn wkv7_pretrain<B: Wkv7Backend>(inputs: Wkv7PretrainForwardInputs<B>) -> Tensor<B, 4> {
    wkv7_pretrain_custom(inputs)
}

/// Convenience wrapper for the StateTuning WKV path.
pub fn wkv7_statetune<B: Wkv7Backend>(inputs: Wkv7StatetuneForwardInputs<B>) -> Tensor<B, 4> {
    wkv7_statetune_custom(inputs)
}

/// Convenience wrapper for the state-passing WKV path.
pub fn wkv7_statepass<B: Wkv7Backend>(
    inputs: Wkv7StatepassForwardInputs<B>,
) -> Wkv7StatepassForwardOutput<B> {
    wkv7_statepass_custom(inputs)
}

/// Computes zero-state RWKV7 WKV with Burn tensor operations.
pub fn wkv7_pretrain_reference<B: Wkv7Backend>(
    inputs: Wkv7PretrainForwardInputs<B>,
) -> Tensor<B, 4> {
    let [batch_size, _context_len, num_heads, head_size] = inputs.receptance.dims();
    let state = Tensor::zeros(
        [batch_size, num_heads, head_size, head_size],
        &inputs.receptance.device(),
    );

    wkv7_reference_with_state(inputs, state).output
}

/// Computes supplied-state RWKV7 WKV with Burn tensor operations.
pub fn wkv7_statetune_reference<B: Wkv7Backend>(
    inputs: Wkv7StatetuneForwardInputs<B>,
) -> Tensor<B, 4> {
    wkv7_reference_with_state(inputs.sequence, inputs.initial_state).output
}

/// Computes state-passing RWKV7 WKV with Burn tensor operations.
pub fn wkv7_statepass_reference<B: Wkv7Backend>(
    inputs: Wkv7StatepassForwardInputs<B>,
) -> Wkv7StatepassForwardOutput<B> {
    wkv7_reference_with_state(inputs.sequence, inputs.initial_state)
}

fn wkv7_reference_with_state<B: Wkv7Backend>(
    inputs: Wkv7PretrainForwardInputs<B>,
    initial_state: Tensor<B, 4>,
) -> Wkv7StatepassForwardOutput<B> {
    let [batch_size, context_len, num_heads, head_size] = inputs.receptance.dims();
    let mut state = initial_state;
    let mut outputs = Vec::with_capacity(context_len);

    for time_index in 0..context_len {
        let range = [
            0..batch_size,
            time_index..(time_index + 1),
            0..num_heads,
            0..head_size,
        ];
        let receptance = inputs
            .receptance
            .clone()
            .slice(range.clone())
            .reshape([batch_size, num_heads, head_size]);
        let decay = (-inputs.weight_decay.clone().slice(range.clone()).exp())
            .exp()
            .reshape([batch_size, num_heads, head_size]);
        let replacement_key = inputs
            .replacement_key
            .clone()
            .slice(range.clone())
            .reshape([batch_size, num_heads, head_size]);
        let value = inputs
            .value
            .clone()
            .slice(range.clone())
            .reshape([batch_size, num_heads, head_size]);
        let removal_key_normalized = inputs
            .removal_key_normalized
            .clone()
            .slice(range.clone())
            .reshape([batch_size, num_heads, head_size]);
        let replacement = inputs
            .replacement
            .clone()
            .slice(range)
            .reshape([batch_size, num_heads, head_size]);

        let state_replacement = (state.clone()
            * removal_key_normalized.clone().unsqueeze_dim::<4>(2))
        .sum_dim(3)
        .reshape([batch_size, num_heads, head_size]);

        state = state * decay.unsqueeze_dim::<4>(2)
            + state_replacement.unsqueeze_dim::<4>(3) * replacement.unsqueeze_dim::<4>(2)
            + value.unsqueeze_dim::<4>(3) * replacement_key.unsqueeze_dim::<4>(2);

        let output = (state.clone() * receptance.unsqueeze_dim::<4>(2))
            .sum_dim(3)
            .reshape([batch_size, 1, num_heads, head_size]);
        outputs.push(output);
    }

    Wkv7StatepassForwardOutput {
        output: Tensor::cat(outputs, 1),
        next_state: state,
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Distribution, Tensor, Tolerance};

    use crate::{
        kernels::train::time_mixer::wkv7::{
            io::{
                Wkv7PretrainForwardInputs,
                Wkv7StatepassForwardInputs,
                Wkv7StatetuneForwardInputs,
            },
            wkv7_pretrain_custom,
            wkv7_pretrain_reference,
            wkv7_statepass_custom,
            wkv7_statepass_reference,
            wkv7_statetune_custom,
            wkv7_statetune_reference,
        },
        test_utils::backend::{TestAutodiffBackend, TestBackend},
    };

    #[test]
    fn forward() {
        let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();
        let inputs = random_sequence::<TestBackend>([2, 16, 2, 8], &device);
        let initial_state =
            Tensor::<TestBackend, 4>::random([2, 2, 8, 8], Distribution::Default, &device);

        let reference = wkv7_pretrain_reference(inputs.clone().into_inputs());
        let custom = wkv7_pretrain_custom(inputs.clone().into_inputs());
        assert_close(reference, custom);

        let statetune_inputs = Wkv7StatetuneForwardInputs {
            initial_state: initial_state.clone(),
            sequence: inputs.clone().into_inputs(),
        };
        let reference = wkv7_statetune_reference(statetune_inputs.clone());
        let custom = wkv7_statetune_custom(statetune_inputs);
        assert_close(reference, custom);

        let statepass_inputs = Wkv7StatepassForwardInputs {
            initial_state,
            sequence: inputs.into_inputs(),
        };
        let reference = wkv7_statepass_reference(statepass_inputs.clone());
        let custom = wkv7_statepass_custom(statepass_inputs);
        assert_close(reference.output, custom.output);
        assert_close(reference.next_state, custom.next_state);
    }

    #[test]
    fn backward() {
        let device: <TestAutodiffBackend as burn::tensor::backend::Backend>::Device =
            Default::default();
        let inputs = random_sequence::<TestAutodiffBackend>([1, 16, 1, 4], &device).require_grad();

        let reference = wkv7_pretrain_reference(inputs.clone().into_inputs()).sum();
        let mut gradients = reference.backward();
        let reference_grads = inputs.clone().remove_grads(&mut gradients);

        let custom = wkv7_pretrain_custom(inputs.clone().into_inputs()).sum();
        let mut gradients = custom.backward();
        let custom_grads = inputs.remove_grads(&mut gradients);

        for (reference, custom) in [
            "receptance",
            "weight_decay",
            "replacement_key",
            "value",
            "removal_key_normalized",
            "replacement",
        ]
        .into_iter()
        .zip(reference_grads.into_iter().zip(custom_grads))
        .map(|(_, grads)| grads)
        {
            reference
                .into_data()
                .convert::<f32>()
                .assert_approx_eq::<f32>(
                    &custom.into_data().convert::<f32>(),
                    Tolerance::absolute(1e-2),
                );
        }
    }

    #[test]
    #[should_panic]
    fn forward_panics_on_wrong_context_len() {
        let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();
        let inputs = random_sequence::<TestBackend>([1, 15, 1, 4], &device);

        let _ = wkv7_pretrain_custom(inputs.into_inputs());
    }

    #[derive(Clone)]
    struct TestInputs<B: super::Wkv7Backend> {
        receptance: Tensor<B, 4>,
        weight_decay: Tensor<B, 4>,
        replacement_key: Tensor<B, 4>,
        value: Tensor<B, 4>,
        removal_key_normalized: Tensor<B, 4>,
        replacement: Tensor<B, 4>,
    }

    impl<B: super::Wkv7Backend> TestInputs<B> {
        fn into_inputs(self) -> Wkv7PretrainForwardInputs<B> {
            Wkv7PretrainForwardInputs {
                receptance: self.receptance,
                weight_decay: self.weight_decay,
                replacement_key: self.replacement_key,
                value: self.value,
                removal_key_normalized: self.removal_key_normalized,
                replacement: self.replacement,
                chunk_len: 16,
            }
        }
    }

    impl<B: super::AutodiffBackend> TestInputs<B> {
        fn require_grad(self) -> Self {
            Self {
                receptance: self.receptance.require_grad(),
                weight_decay: self.weight_decay.require_grad(),
                replacement_key: self.replacement_key.require_grad(),
                value: self.value.require_grad(),
                removal_key_normalized: self.removal_key_normalized.require_grad(),
                replacement: self.replacement.require_grad(),
            }
        }

        fn remove_grads(self, gradients: &mut B::Gradients) -> Vec<Tensor<B::InnerBackend, 4>> {
            vec![
                self.receptance.grad_remove(gradients).unwrap(),
                self.weight_decay.grad_remove(gradients).unwrap(),
                self.replacement_key.grad_remove(gradients).unwrap(),
                self.value.grad_remove(gradients).unwrap(),
                self.removal_key_normalized.grad_remove(gradients).unwrap(),
                self.replacement.grad_remove(gradients).unwrap(),
            ]
        }
    }

    fn random_sequence<B: super::Wkv7Backend>(
        shape: [usize; 4],
        device: &B::Device,
    ) -> TestInputs<B> {
        TestInputs {
            receptance: Tensor::random(shape, Distribution::Default, device),
            weight_decay: Tensor::random(shape, Distribution::Default, device),
            replacement_key: Tensor::random(shape, Distribution::Default, device),
            value: Tensor::random(shape, Distribution::Default, device),
            removal_key_normalized: Tensor::random(shape, Distribution::Default, device),
            replacement: Tensor::random(shape, Distribution::Default, device),
        }
    }

    fn assert_close<B: super::Wkv7Backend>(reference: Tensor<B, 4>, custom: Tensor<B, 4>) {
        reference
            .into_data()
            .convert::<f32>()
            .assert_approx_eq::<f32>(
                &custom.into_data().convert::<f32>(),
                Tolerance::absolute(1e-2),
            );
    }
}
