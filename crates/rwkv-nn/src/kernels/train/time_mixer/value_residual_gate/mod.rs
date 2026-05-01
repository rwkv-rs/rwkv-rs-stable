mod backward;
mod forward;
/// Input containers for the value residual gate kernel.
pub mod io;
mod kernel;

use burn::tensor::{Tensor, TensorPrimitive, activation::sigmoid, ops::FloatTensor};
use burn_cubecl::{
    CubeBackend,
    CubeElement,
    CubeRuntime,
    FloatElement,
    IntElement,
    element::BoolElement,
};

use crate::kernels::train::time_mixer::value_residual_gate::io::{
    ValueResidualGateForwardInputs,
    ValueResidualGateForwardPrimitiveInputs,
};

/// Backend primitive capability for the fused value residual gate.
pub trait ValueResidualGateBackend: burn::tensor::backend::Backend {
    /// Runs the RWKV7 value residual gate as one fused primitive operation.
    fn fused_value_residual_gate(
        inputs: ValueResidualGateForwardPrimitiveInputs<Self>,
    ) -> FloatTensor<Self>;
}

/// Autodiff backend marker for the value residual gate.
pub trait AutodiffBackend:
    ValueResidualGateBackend + burn::tensor::backend::AutodiffBackend
{
}

impl<B> AutodiffBackend for B where
    B: ValueResidualGateBackend + burn::tensor::backend::AutodiffBackend
{
}

impl<R, F, I, BT> ValueResidualGateBackend for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
{
    fn fused_value_residual_gate(
        inputs: ValueResidualGateForwardPrimitiveInputs<Self>,
    ) -> FloatTensor<Self> {
        assert!(inputs.value.is_contiguous(), "value must be contiguous");
        assert!(
            inputs.value_from_first_cell.is_contiguous(),
            "value_from_first_cell must be contiguous"
        );
        assert!(
            inputs.gate_base.is_contiguous(),
            "gate_base must be contiguous"
        );
        assert!(
            inputs.gate_input.is_contiguous(),
            "gate_input must be contiguous"
        );

        forward::fused_value_residual_gate::<R, F, I, BT>(inputs)
    }
}

/// Runs the fused RWKV7 value residual gate after validating the public input contract.
///
/// `value`, `value_from_first_cell`, and `gate_input` must be contiguous and shaped
/// `[batch_size, context_len, embedded_dim]`. `gate_base` must be contiguous and shaped
/// `[embedded_dim]`.
///
/// Mathematically:
/// `gate[b, t, e] = sigmoid(gate_base[e] + gate_input[b, t, e])`;
/// `output[b, t, e] = value[b, t, e] + (value_from_first_cell[b, t, e] - value[b, t, e]) *
/// gate[b, t, e]`.
///
/// This ports the RWKV-LM `tmix_vres_gate` fast path using repository terminology. The custom
/// path fuses the broadcast add, sigmoid, and residual blend into one elementwise kernel and keeps
/// the gate-base broadcast as index math. Backward uses CubeCL kernels for the elementwise
/// gradients and the gate-base reduction across `[batch_size, context_len]`.
pub fn value_residual_gate_custom<B: ValueResidualGateBackend>(
    inputs: ValueResidualGateForwardInputs<B>,
) -> Tensor<B, 3> {
    inputs.check().unwrap();
    let output = B::fused_value_residual_gate(inputs.to_primitive());

    Tensor::from_primitive(TensorPrimitive::Float(output))
}

/// Computes the value residual gate with regular Burn tensor operations.
///
/// This is the semantic reference for correctness and gradient tests. Burn fusion may simplify the
/// generic expression graph, while the custom path exposes an explicit CubeCL primitive with
/// autotuned vector width.
pub fn value_residual_gate_reference<B: ValueResidualGateBackend>(
    inputs: ValueResidualGateForwardInputs<B>,
) -> Tensor<B, 3> {
    let gate =
        sigmoid(inputs.gate_input + inputs.gate_base.unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0));

    inputs.value.clone() + (inputs.value_from_first_cell - inputs.value) * gate
}

/// Convenience wrapper for the fused value residual gate path.
pub fn value_residual_gate<B: ValueResidualGateBackend>(
    value: Tensor<B, 3>,
    value_from_first_cell: Tensor<B, 3>,
    gate_base: Tensor<B, 1>,
    gate_input: Tensor<B, 3>,
) -> Tensor<B, 3> {
    value_residual_gate_custom(ValueResidualGateForwardInputs {
        value,
        value_from_first_cell,
        gate_base,
        gate_input,
    })
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Distribution, Tensor, Tolerance};

    use crate::{
        kernels::train::time_mixer::value_residual_gate::{
            io::ValueResidualGateForwardInputs,
            value_residual_gate_custom,
            value_residual_gate_reference,
        },
        test_utils::backend::{TestAutodiffBackend, TestAutodiffDevice, TestBackend, TestDevice},
    };

    #[test]
    fn forward() {
        let device: TestDevice = Default::default();

        for shape in [[2, 8, 32], [1, 3, 17]] {
            let value = Tensor::<TestBackend, 3>::random(shape, Distribution::Default, &device);
            let value_from_first_cell =
                Tensor::<TestBackend, 3>::random(shape, Distribution::Default, &device);
            let gate_base =
                Tensor::<TestBackend, 1>::random([shape[2]], Distribution::Default, &device);
            let gate_input =
                Tensor::<TestBackend, 3>::random(shape, Distribution::Default, &device);
            let inputs = ValueResidualGateForwardInputs {
                value,
                value_from_first_cell,
                gate_base,
                gate_input,
            };

            let reference = value_residual_gate_reference(inputs.clone())
                .into_data()
                .convert::<f32>();
            let custom = value_residual_gate_custom(inputs)
                .into_data()
                .convert::<f32>();

            reference.assert_approx_eq::<f32>(&custom, Tolerance::default());
        }
    }

    #[test]
    fn backward() {
        let device: TestAutodiffDevice = Default::default();

        for shape in [[2, 8, 32], [1, 3, 17]] {
            let value =
                Tensor::<TestAutodiffBackend, 3>::random(shape, Distribution::Default, &device)
                    .require_grad();
            let value_from_first_cell =
                Tensor::<TestAutodiffBackend, 3>::random(shape, Distribution::Default, &device)
                    .require_grad();
            let gate_base = Tensor::<TestAutodiffBackend, 1>::random(
                [shape[2]],
                Distribution::Default,
                &device,
            )
            .require_grad();
            let gate_input =
                Tensor::<TestAutodiffBackend, 3>::random(shape, Distribution::Default, &device)
                    .require_grad();

            let reference = value_residual_gate_reference(ValueResidualGateForwardInputs {
                value: value.clone(),
                value_from_first_cell: value_from_first_cell.clone(),
                gate_base: gate_base.clone(),
                gate_input: gate_input.clone(),
            });
            let mut gradients = reference.backward();
            let value_grad_ref = value.grad_remove(&mut gradients).unwrap();
            let value_from_first_cell_grad_ref =
                value_from_first_cell.grad_remove(&mut gradients).unwrap();
            let gate_base_grad_ref = gate_base.grad_remove(&mut gradients).unwrap();
            let gate_input_grad_ref = gate_input.grad_remove(&mut gradients).unwrap();

            let value_custom = value.detach().require_grad();
            let value_from_first_cell_custom = value_from_first_cell.detach().require_grad();
            let gate_base_custom = gate_base.detach().require_grad();
            let gate_input_custom = gate_input.detach().require_grad();
            let custom = value_residual_gate_custom(ValueResidualGateForwardInputs {
                value: value_custom.clone(),
                value_from_first_cell: value_from_first_cell_custom.clone(),
                gate_base: gate_base_custom.clone(),
                gate_input: gate_input_custom.clone(),
            });
            let mut gradients = custom.backward();
            let value_grad_custom = value_custom.grad_remove(&mut gradients).unwrap();
            let value_from_first_cell_grad_custom = value_from_first_cell_custom
                .grad_remove(&mut gradients)
                .unwrap();
            let gate_base_grad_custom = gate_base_custom.grad_remove(&mut gradients).unwrap();
            let gate_input_grad_custom = gate_input_custom.grad_remove(&mut gradients).unwrap();

            value_grad_ref
                .into_data()
                .convert::<f32>()
                .assert_approx_eq::<f32>(
                    &value_grad_custom.into_data().convert::<f32>(),
                    Tolerance::default(),
                );
            value_from_first_cell_grad_ref
                .into_data()
                .convert::<f32>()
                .assert_approx_eq::<f32>(
                    &value_from_first_cell_grad_custom
                        .into_data()
                        .convert::<f32>(),
                    Tolerance::default(),
                );
            gate_base_grad_ref
                .into_data()
                .convert::<f32>()
                .assert_approx_eq::<f32>(
                    &gate_base_grad_custom.into_data().convert::<f32>(),
                    Tolerance::default(),
                );
            gate_input_grad_ref
                .into_data()
                .convert::<f32>()
                .assert_approx_eq::<f32>(
                    &gate_input_grad_custom.into_data().convert::<f32>(),
                    Tolerance::default(),
                );
        }
    }
}
