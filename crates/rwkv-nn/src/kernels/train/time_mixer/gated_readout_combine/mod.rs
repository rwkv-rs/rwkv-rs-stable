mod forward;
/// Input containers for the gated-readout combine kernel.
pub mod io;
mod kernel;

use burn::{
    backend::autodiff::{Autodiff, checkpoint::strategy::CheckpointStrategy},
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
    time_mixer::gated_readout_combine::io::{
        GatedReadoutCombineForwardInputs,
        GatedReadoutCombineForwardPrimitiveInputs,
    },
};

/// Backend primitive capability for the gated-readout combine operation.
pub trait GatedReadoutCombineBackend: burn::tensor::backend::Backend {
    /// Runs the gated-readout combine primitive.
    fn fused_gated_readout_combine(
        inputs: GatedReadoutCombineForwardPrimitiveInputs<Self>,
    ) -> FloatTensor<Self>;
}

impl<R, F, I, BT> GatedReadoutCombineBackend for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
{
    fn fused_gated_readout_combine(
        inputs: GatedReadoutCombineForwardPrimitiveInputs<Self>,
    ) -> FloatTensor<Self> {
        assert_linear_readable("wkv_output", &inputs.wkv_output);
        assert_linear_readable("norm_gamma", &inputs.norm_gamma);
        assert_linear_readable("norm_beta", &inputs.norm_beta);
        assert_linear_readable("gate", &inputs.gate);
        assert_linear_readable("receptance", &inputs.receptance);
        assert_linear_readable("replacement_key", &inputs.replacement_key);
        assert_linear_readable("value", &inputs.value);
        assert_linear_readable("bonus", &inputs.bonus);

        forward::fused_gated_readout_combine::<R, F, I, BT>(inputs)
    }
}

impl<B, C> GatedReadoutCombineBackend for Autodiff<B, C>
where
    B: GatedReadoutCombineBackend,
    C: CheckpointStrategy,
{
    fn fused_gated_readout_combine(
        inputs: GatedReadoutCombineForwardPrimitiveInputs<Self>,
    ) -> FloatTensor<Self> {
        let wkv_output: Tensor<Self, 4> =
            Tensor::from_primitive(TensorPrimitive::<Self>::Float(inputs.wkv_output));
        let norm_gamma: Tensor<Self, 1> =
            Tensor::from_primitive(TensorPrimitive::<Self>::Float(inputs.norm_gamma));
        let norm_beta: Tensor<Self, 1> =
            Tensor::from_primitive(TensorPrimitive::<Self>::Float(inputs.norm_beta));
        let gate: Tensor<Self, 3> =
            Tensor::from_primitive(TensorPrimitive::<Self>::Float(inputs.gate));
        let receptance: Tensor<Self, 4> =
            Tensor::from_primitive(TensorPrimitive::<Self>::Float(inputs.receptance));
        let replacement_key: Tensor<Self, 4> =
            Tensor::from_primitive(TensorPrimitive::<Self>::Float(inputs.replacement_key));
        let value: Tensor<Self, 4> =
            Tensor::from_primitive(TensorPrimitive::<Self>::Float(inputs.value));
        let bonus: Tensor<Self, 2> =
            Tensor::from_primitive(TensorPrimitive::<Self>::Float(inputs.bonus));

        gated_readout_combine_reference(GatedReadoutCombineForwardInputs {
            wkv_output,
            norm_gamma,
            norm_beta,
            norm_epsilon: inputs.norm_epsilon,
            gate,
            receptance,
            replacement_key,
            value,
            bonus,
        })
        .into_primitive()
        .tensor()
    }
}

/// Runs the gated-readout combine after validating the public input contract.
pub fn gated_readout_combine<B: GatedReadoutCombineBackend>(
    inputs: GatedReadoutCombineForwardInputs<B>,
) -> Tensor<B, 3> {
    inputs.check().unwrap();
    let output = B::fused_gated_readout_combine(inputs.to_primitive());

    Tensor::from_primitive(TensorPrimitive::Float(output))
}

/// Computes gated-readout combine with regular Burn tensor operations.
pub fn gated_readout_combine_reference<B: GatedReadoutCombineBackend>(
    inputs: GatedReadoutCombineForwardInputs<B>,
) -> Tensor<B, 3> {
    let [batch_size, context_len, num_heads, head_size] = inputs.wkv_output.dims();
    let embedded_dim = num_heads * head_size;
    let rows = batch_size * context_len;

    let grouped = inputs.wkv_output.reshape([rows, num_heads, head_size]);
    let mean = grouped.clone().sum_dim(2) / head_size as f64;
    let centered = grouped.sub(mean);
    let var = centered.clone().square().sum_dim(2) / head_size as f64;
    let normalized = centered
        .div(var.add_scalar(inputs.norm_epsilon).sqrt())
        .reshape([batch_size, context_len, embedded_dim])
        * inputs.norm_gamma.reshape([1, 1, embedded_dim])
        + inputs.norm_beta.reshape([1, 1, embedded_dim]);

    let bonus: Tensor<B, 4> =
        (inputs.receptance * inputs.replacement_key * inputs.bonus.unsqueeze_dims(&[0, 1]))
            .sum_dim(3)
            * inputs.value;
    let bonus = bonus.reshape([batch_size, context_len, embedded_dim]);

    (normalized + bonus) * inputs.gate
}
