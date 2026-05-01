mod backward;
mod forward;
/// Input containers for the RWKV7 pretrain channel mixer kernel.
pub mod io;
mod kernel;

use burn::tensor::{Tensor, TensorPrimitive, activation::relu, ops::FloatTensor};
use burn_cubecl::{
    CubeBackend,
    CubeElement,
    CubeRuntime,
    FloatElement,
    IntElement,
    element::BoolElement,
};

use crate::kernels::train::channel_mixer::io::{
    ChannelMixerForwardInputs,
    ChannelMixerForwardPrimitiveInputs,
};

/// Backend primitive capability for the fused RWKV7 pretrain channel mixer.
pub trait ChannelMixerBackend: burn::tensor::backend::Backend {
    /// Runs the channel mixer layer as one primitive operation.
    fn fused_channel_mixer(inputs: ChannelMixerForwardPrimitiveInputs<Self>) -> FloatTensor<Self>;
}

/// Autodiff backend marker for the channel mixer kernel.
pub trait AutodiffBackend: ChannelMixerBackend + burn::tensor::backend::AutodiffBackend {}

impl<B> AutodiffBackend for B where B: ChannelMixerBackend + burn::tensor::backend::AutodiffBackend {}

impl<R, F, I, BT> ChannelMixerBackend for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
{
    fn fused_channel_mixer(inputs: ChannelMixerForwardPrimitiveInputs<Self>) -> FloatTensor<Self> {
        assert!(
            inputs.embedded_context.is_contiguous(),
            "embedded_context must be contiguous"
        );
        assert!(
            inputs.key_scale.is_contiguous(),
            "key_scale must be contiguous"
        );
        assert!(
            inputs.key_weight.is_contiguous(),
            "key_weight must be contiguous"
        );
        assert!(
            inputs.value_weight.is_contiguous(),
            "value_weight must be contiguous"
        );

        forward::fused_channel_mixer::<R, F, I, BT>(inputs)
    }
}

/// Runs the fused RWKV7 pretrain channel mixer after validating the public input contract.
///
/// `embedded_context` must be contiguous and shaped `[batch_size, context_len, embedded_dim]`.
/// `key_scale` must be contiguous and shaped `[embedded_dim]`. `key_weight` must be contiguous
/// and shaped `[expanded_dim, embedded_dim]`. `value_weight` must be contiguous and shaped
/// `[embedded_dim, expanded_dim]`.
///
/// For each token, the operation first computes the token-shift channel mix:
/// `previous = 0` for `time_index == 0`, otherwise the previous token from the same batch;
/// `key_input = embedded_context + (previous - embedded_context) * key_scale`.
/// It then computes `activated_key = relu(key_input @ key_weight.T)^2` and
/// `output = activated_key @ value_weight.T`.
///
/// This ports the RWKV-LM `cmix` fast path using repository terminology. The custom primitive
/// fuses the token-shift mix and ReLU-square elementwise stages while using backend matmul
/// primitives for the two projections.
pub fn channel_mixer_custom<B: ChannelMixerBackend>(
    inputs: ChannelMixerForwardInputs<B>,
) -> Tensor<B, 3> {
    inputs.check().unwrap();
    let output = B::fused_channel_mixer(inputs.to_primitive());

    Tensor::from_primitive(TensorPrimitive::Float(output))
}

/// Computes the RWKV7 pretrain channel mixer with regular Burn tensor operations.
///
/// This is the semantic reference for correctness and gradient tests. Burn fusion may simplify
/// parts of the expression graph, while the custom path exposes explicit fused elementwise stages
/// around the backend matmul operations.
pub fn channel_mixer_reference<B: burn::tensor::backend::Backend>(
    inputs: ChannelMixerForwardInputs<B>,
) -> Tensor<B, 3> {
    let [batch_size, context_len, embedded_dim] = inputs.embedded_context.dims();
    let device = inputs.embedded_context.device();

    if context_len == 0 {
        return Tensor::<B, 3>::zeros([batch_size, 0, embedded_dim], &device);
    }

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
    let key_input = inputs.embedded_context
        + token_shifted_diff * inputs.key_scale.unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0);
    let rows = batch_size * context_len;
    let key_input = key_input.reshape([rows, embedded_dim]);
    let activated_key = relu(key_input.matmul(inputs.key_weight.transpose())).powf_scalar(2.0);
    let output = activated_key.matmul(inputs.value_weight.transpose());

    output.reshape([batch_size, context_len, embedded_dim])
}

/// Convenience wrapper for the fused pretrain channel mixer path.
pub fn channel_mixer<B: ChannelMixerBackend>(
    embedded_context: Tensor<B, 3>,
    key_scale: Tensor<B, 1>,
    key_weight: Tensor<B, 2>,
    value_weight: Tensor<B, 2>,
) -> Tensor<B, 3> {
    channel_mixer_custom(ChannelMixerForwardInputs {
        embedded_context,
        key_scale,
        key_weight,
        value_weight,
    })
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::autodiff::grads::Gradients,
        tensor::{Distribution, Tensor, Tolerance},
    };

    use crate::{
        kernels::train::channel_mixer::{
            channel_mixer_custom,
            channel_mixer_reference,
            io::ChannelMixerForwardInputs,
        },
        test_utils::backend::{TestAutodiffBackend, TestAutodiffDevice, TestBackend, TestDevice},
    };

    #[test]
    fn forward() {
        let device: TestDevice = Default::default();

        for shape in [[2, 8, 32], [1, 3, 17], [2, 0, 32]] {
            let inputs = random_inputs::<TestBackend>(shape[0], shape[1], shape[2], &device);
            if shape[1] == 0 {
                assert_eq!(channel_mixer_custom(inputs).dims(), shape);
                continue;
            }

            let reference = channel_mixer_reference(inputs.clone())
                .into_data()
                .convert::<f32>();
            let custom = channel_mixer_custom(inputs).into_data().convert::<f32>();

            reference.assert_approx_eq::<f32>(&custom, Tolerance::default());
        }
    }

    #[test]
    fn backward() {
        let device: TestAutodiffDevice = Default::default();

        for shape in [[2, 8, 32], [1, 3, 17], [2, 1, 32]] {
            assert_backward_close(shape[0], shape[1], shape[2], &device);
        }
    }

    fn assert_backward_close(
        batch_size: usize,
        context_len: usize,
        embedded_dim: usize,
        device: &TestAutodiffDevice,
    ) {
        let base_inputs =
            random_inputs::<TestAutodiffBackend>(batch_size, context_len, embedded_dim, device);
        let inputs = require_grad(base_inputs.clone());
        let reference = channel_mixer_reference(inputs.clone()).sum();
        let mut gradients = reference.backward();
        let reference_grads = remove_grads(inputs, &mut gradients);

        let inputs = require_grad(detach_inputs(base_inputs));
        let custom = channel_mixer_custom(inputs.clone()).sum();
        let mut gradients = custom.backward();
        let custom_grads = remove_grads(inputs, &mut gradients);

        assert_grads_close(reference_grads, custom_grads);
    }

    #[test]
    #[should_panic(expected = "ShapeMismatch")]
    fn rejects_wrong_key_scale_shape() {
        let device: TestDevice = Default::default();
        let mut inputs = random_inputs::<TestBackend>(2, 8, 32, &device);
        inputs.key_scale = Tensor::<TestBackend, 1>::random([31], Distribution::Default, &device);

        channel_mixer_custom(inputs);
    }

    #[test]
    #[should_panic(expected = "AxisMismatch")]
    fn rejects_wrong_key_weight_shape() {
        let device: TestDevice = Default::default();
        let mut inputs = random_inputs::<TestBackend>(2, 8, 32, &device);
        inputs.key_weight =
            Tensor::<TestBackend, 2>::random([128, 31], Distribution::Default, &device);

        channel_mixer_custom(inputs);
    }

    #[test]
    #[should_panic(expected = "AxisMismatch")]
    fn rejects_wrong_value_weight_shape() {
        let device: TestDevice = Default::default();
        let mut inputs = random_inputs::<TestBackend>(2, 8, 32, &device);
        inputs.value_weight =
            Tensor::<TestBackend, 2>::random([32, 127], Distribution::Default, &device);

        channel_mixer_custom(inputs);
    }

    fn random_inputs<B: burn::tensor::backend::Backend>(
        batch_size: usize,
        context_len: usize,
        embedded_dim: usize,
        device: &B::Device,
    ) -> ChannelMixerForwardInputs<B> {
        let expanded_dim = embedded_dim * 4;

        ChannelMixerForwardInputs {
            embedded_context: Tensor::<B, 3>::random(
                [batch_size, context_len, embedded_dim],
                Distribution::Default,
                device,
            ),
            key_scale: Tensor::<B, 1>::random([embedded_dim], Distribution::Default, device),
            key_weight: Tensor::<B, 2>::random(
                [expanded_dim, embedded_dim],
                Distribution::Default,
                device,
            ),
            value_weight: Tensor::<B, 2>::random(
                [embedded_dim, expanded_dim],
                Distribution::Default,
                device,
            ),
        }
    }

    fn require_grad(
        inputs: ChannelMixerForwardInputs<TestAutodiffBackend>,
    ) -> ChannelMixerForwardInputs<TestAutodiffBackend> {
        ChannelMixerForwardInputs {
            embedded_context: inputs.embedded_context.require_grad(),
            key_scale: inputs.key_scale.require_grad(),
            key_weight: inputs.key_weight.require_grad(),
            value_weight: inputs.value_weight.require_grad(),
        }
    }

    fn detach_inputs(
        inputs: ChannelMixerForwardInputs<TestAutodiffBackend>,
    ) -> ChannelMixerForwardInputs<TestAutodiffBackend> {
        ChannelMixerForwardInputs {
            embedded_context: inputs.embedded_context.detach(),
            key_scale: inputs.key_scale.detach(),
            key_weight: inputs.key_weight.detach(),
            value_weight: inputs.value_weight.detach(),
        }
    }

    fn remove_grads(
        inputs: ChannelMixerForwardInputs<TestAutodiffBackend>,
        gradients: &mut Gradients,
    ) -> ChannelMixerGrads<TestBackend> {
        ChannelMixerGrads {
            embedded_context: inputs.embedded_context.grad_remove(gradients).unwrap(),
            key_scale: inputs.key_scale.grad_remove(gradients).unwrap(),
            key_weight: inputs.key_weight.grad_remove(gradients).unwrap(),
            value_weight: inputs.value_weight.grad_remove(gradients).unwrap(),
        }
    }

    struct ChannelMixerGrads<B: burn::tensor::backend::Backend> {
        embedded_context: Tensor<B, 3>,
        key_scale: Tensor<B, 1>,
        key_weight: Tensor<B, 2>,
        value_weight: Tensor<B, 2>,
    }

    fn assert_grads_close(
        reference: ChannelMixerGrads<TestBackend>,
        custom: ChannelMixerGrads<TestBackend>,
    ) {
        reference
            .embedded_context
            .into_data()
            .convert::<f32>()
            .assert_approx_eq::<f32>(
                &custom.embedded_context.into_data().convert::<f32>(),
                Tolerance::default(),
            );
        reference
            .key_scale
            .into_data()
            .convert::<f32>()
            .assert_approx_eq::<f32>(
                &custom.key_scale.into_data().convert::<f32>(),
                Tolerance::default(),
            );
        reference
            .key_weight
            .into_data()
            .convert::<f32>()
            .assert_approx_eq::<f32>(
                &custom.key_weight.into_data().convert::<f32>(),
                Tolerance::default(),
            );
        reference
            .value_weight
            .into_data()
            .convert::<f32>()
            .assert_approx_eq::<f32>(
                &custom.value_weight.into_data().convert::<f32>(),
                Tolerance::default(),
            );
    }
}
