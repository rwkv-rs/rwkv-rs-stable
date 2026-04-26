mod backward;
mod forward;
/// Input and output containers for token-shift difference.
pub mod io;
mod kernel;

use burn::{
    prelude::Int,
    tensor::{IndexingUpdateOp, Tensor, TensorPrimitive},
};
use burn_cubecl::{
    CubeBackend,
    CubeElement,
    CubeRuntime,
    FloatElement,
    IntElement,
    element::BoolElement,
};

use crate::kernels::token_shift_diff::io::{
    TokenShiftDiffForwardInputs,
    TokenShiftDiffForwardOutput,
    TokenShiftDiffForwardPrimitiveInputs,
    TokenShiftDiffForwardPrimitiveOutput,
};

/// Backend primitive capability for fused token-shift difference.
pub trait TokenShiftDiffBackend: burn::tensor::backend::Backend {
    /// Runs token-shift difference and state update as one fused primitive operation.
    fn fused_token_shift_diff(
        inputs: TokenShiftDiffForwardPrimitiveInputs<Self>,
    ) -> TokenShiftDiffForwardPrimitiveOutput<Self>;
}

/// Autodiff backend marker for token-shift difference.
pub trait AutodiffBackend: TokenShiftDiffBackend + burn::tensor::backend::AutodiffBackend {}

impl<B> AutodiffBackend for B where B: TokenShiftDiffBackend + burn::tensor::backend::AutodiffBackend
{}

impl<R, F, I, BT> TokenShiftDiffBackend for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
{
    fn fused_token_shift_diff(
        inputs: TokenShiftDiffForwardPrimitiveInputs<Self>,
    ) -> TokenShiftDiffForwardPrimitiveOutput<Self> {
        assert!(
            inputs.embedded_context.is_contiguous(),
            "embedded_context must be contiguous"
        );
        assert!(
            inputs.embedded_token_shift.is_contiguous(),
            "embedded_token_shift must be contiguous"
        );
        assert!(
            inputs.batch_ids.is_contiguous(),
            "batch_ids must be contiguous"
        );

        forward::fused_token_shift_diff::<R, F, I, BT>(inputs)
    }
}

/// Runs the fused token-shift difference kernel after validating the public input contract.
///
/// `embedded_context` must be shaped `[batch_size, context_len, embedded_dim]`,
/// `embedded_token_shift` must be shaped `[full_batch_size, embedded_dim]`, and `batch_ids`
/// must be shaped `[batch_size]`. `batch_ids` values are expected to be valid, unique rows in
/// the full token-shift state.
///
/// The operation is:
/// `prev[b, 0, e] = embedded_token_shift[batch_ids[b], e]`;
/// `prev[b, t, e] = embedded_context[b, t - 1, e]` for `t > 0`;
/// `token_shifted_diff = prev - embedded_context`;
/// `next_token_shift[batch_ids[b], e] = embedded_context[b, context_len - 1, e]`.
/// Rows not selected by `batch_ids` keep their previous state. For `context_len == 0`, the
/// difference output is empty and the state is returned unchanged.
pub fn token_shift_diff_custom<B: TokenShiftDiffBackend>(
    inputs: TokenShiftDiffForwardInputs<B>,
) -> TokenShiftDiffForwardOutput<B> {
    inputs.check().unwrap();

    let context_len = inputs.embedded_context.dims()[1];
    let output = if context_len == 0 {
        TokenShiftDiffForwardPrimitiveOutput {
            token_shifted_diff: inputs.embedded_context.clone().into_primitive().tensor(),
            next_token_shift: inputs
                .embedded_token_shift
                .clone()
                .into_primitive()
                .tensor(),
        }
    } else {
        B::fused_token_shift_diff(inputs.to_primitive())
    };

    TokenShiftDiffForwardOutput {
        token_shifted_diff: Tensor::from_primitive(TensorPrimitive::Float(
            output.token_shifted_diff,
        )),
        next_token_shift: Tensor::from_primitive(TensorPrimitive::Float(output.next_token_shift)),
    }
}

/// Computes token-shift difference with regular tensor operations as the semantic reference.
///
/// This reference expresses the same shift, subtraction, and selected-row state update with Burn
/// tensor operations. Burn fusion may simplify parts of this graph, while the custom path keeps the
/// state read, per-time difference, and final selected-row write in one primitive kernel.
pub fn token_shift_diff_reference<B: TokenShiftDiffBackend>(
    inputs: TokenShiftDiffForwardInputs<B>,
) -> TokenShiftDiffForwardOutput<B> {
    let [batch_size, context_len, embedded_dim] = inputs.embedded_context.dims();

    if context_len == 0 {
        return TokenShiftDiffForwardOutput {
            token_shifted_diff: inputs.embedded_context,
            next_token_shift: inputs.embedded_token_shift,
        };
    }

    let active_token_shift = inputs
        .embedded_token_shift
        .clone()
        .select(0, inputs.batch_ids.clone());
    let shifted = if context_len == 1 {
        active_token_shift.unsqueeze_dim(1)
    } else {
        Tensor::cat(
            vec![
                active_token_shift.unsqueeze_dim(1),
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

    TokenShiftDiffForwardOutput {
        token_shifted_diff,
        next_token_shift: next_token_shift_reference(
            inputs.embedded_context,
            inputs.embedded_token_shift,
            inputs.batch_ids,
        ),
    }
}

/// Convenience wrapper for the fused token-shift difference path.
pub fn token_shift_diff<B: TokenShiftDiffBackend>(
    embedded_context: Tensor<B, 3>,
    embedded_token_shift: Tensor<B, 2>,
    batch_ids: Tensor<B, 1, Int>,
) -> TokenShiftDiffForwardOutput<B> {
    token_shift_diff_custom(TokenShiftDiffForwardInputs {
        embedded_context,
        embedded_token_shift,
        batch_ids,
    })
}

fn next_token_shift_reference<B: TokenShiftDiffBackend>(
    embedded_context: Tensor<B, 3>,
    embedded_token_shift: Tensor<B, 2>,
    batch_ids: Tensor<B, 1, Int>,
) -> Tensor<B, 2> {
    let [batch_size, context_len, embedded_dim] = embedded_context.dims();
    let [full_batch_size, _] = embedded_token_shift.dims();
    let device = embedded_context.device();

    if context_len == 0 {
        return embedded_token_shift;
    }

    let last_token = embedded_context
        .slice([
            0..batch_size,
            (context_len - 1)..context_len,
            0..embedded_dim,
        ])
        .squeeze_dim(1);
    let selected_rows = Tensor::<B, 1>::zeros([full_batch_size], &device).select_assign(
        0,
        batch_ids.clone(),
        Tensor::ones([batch_size], &device),
        IndexingUpdateOp::Add,
    );

    (embedded_token_shift.clone()
        * (Tensor::ones([full_batch_size], &device) - selected_rows).unsqueeze_dim(1))
    .select_assign(0, batch_ids, last_token, IndexingUpdateOp::Add)
}

#[cfg(test)]
mod tests {
    use burn::{
        prelude::Int,
        tensor::{Distribution, Tensor, TensorData, Tolerance},
    };

    use crate::{
        kernels::token_shift_diff::{
            io::TokenShiftDiffForwardInputs,
            token_shift_diff_custom,
            token_shift_diff_reference,
        },
        test_utils::backend::{TestAutodiffBackend, TestBackend},
    };

    #[test]
    fn forward() {
        let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();

        for (batch_size, context_len, embedded_dim, full_batch_size, ids) in [
            (1, 1, 8, 1, vec![0]),
            (1, 4, 8, 3, vec![2]),
            (3, 5, 16, 8, vec![6, 1, 4]),
            (2, 0, 8, 4, vec![3, 1]),
        ] {
            let inputs = TokenShiftDiffForwardInputs {
                embedded_context: Tensor::<TestBackend, 3>::random(
                    [batch_size, context_len, embedded_dim],
                    Distribution::Default,
                    &device,
                ),
                embedded_token_shift: Tensor::<TestBackend, 2>::random(
                    [full_batch_size, embedded_dim],
                    Distribution::Default,
                    &device,
                ),
                batch_ids: Tensor::<TestBackend, 1, Int>::from_ints(
                    TensorData::new(ids, [batch_size]),
                    &device,
                ),
            };

            let reference = token_shift_diff_reference(inputs.clone());
            let custom = token_shift_diff_custom(inputs);

            assert_eq!(
                reference.token_shifted_diff.dims(),
                custom.token_shifted_diff.dims()
            );
            if context_len > 0 {
                reference
                    .token_shifted_diff
                    .into_data()
                    .convert::<f32>()
                    .assert_approx_eq::<f32>(
                        &custom.token_shifted_diff.into_data().convert::<f32>(),
                        Tolerance::default(),
                    );
            }
            reference
                .next_token_shift
                .into_data()
                .convert::<f32>()
                .assert_approx_eq::<f32>(
                    &custom.next_token_shift.into_data().convert::<f32>(),
                    Tolerance::default(),
                );
        }
    }

    #[test]
    fn backward() {
        let device: <TestAutodiffBackend as burn::tensor::backend::Backend>::Device =
            Default::default();
        let batch_ids = Tensor::<TestAutodiffBackend, 1, Int>::from_ints([4, 1, 6], &device);

        let embedded_context =
            Tensor::<TestAutodiffBackend, 3>::random([3, 5, 16], Distribution::Default, &device)
                .require_grad();
        let embedded_token_shift =
            Tensor::<TestAutodiffBackend, 2>::random([8, 16], Distribution::Default, &device)
                .require_grad();

        let reference = token_shift_diff_reference(TokenShiftDiffForwardInputs {
            embedded_context: embedded_context.clone(),
            embedded_token_shift: embedded_token_shift.clone(),
            batch_ids: batch_ids.clone(),
        });
        let reference = reference.token_shifted_diff.sum() + reference.next_token_shift.sum();
        let mut gradients = reference.backward();
        let embedded_context_grad_ref = embedded_context.grad_remove(&mut gradients).unwrap();
        let embedded_token_shift_grad_ref =
            embedded_token_shift.grad_remove(&mut gradients).unwrap();

        let embedded_context_custom = embedded_context.detach().require_grad();
        let embedded_token_shift_custom = embedded_token_shift.detach().require_grad();
        let custom = token_shift_diff_custom(TokenShiftDiffForwardInputs {
            embedded_context: embedded_context_custom.clone(),
            embedded_token_shift: embedded_token_shift_custom.clone(),
            batch_ids,
        });
        let custom = custom.token_shifted_diff.sum() + custom.next_token_shift.sum();
        let mut gradients = custom.backward();
        let embedded_context_grad_custom =
            embedded_context_custom.grad_remove(&mut gradients).unwrap();
        let embedded_token_shift_grad_custom = embedded_token_shift_custom
            .grad_remove(&mut gradients)
            .unwrap();

        embedded_context_grad_ref
            .into_data()
            .convert::<f32>()
            .assert_approx_eq::<f32>(
                &embedded_context_grad_custom.into_data().convert::<f32>(),
                Tolerance::default(),
            );
        embedded_token_shift_grad_ref
            .into_data()
            .convert::<f32>()
            .assert_approx_eq::<f32>(
                &embedded_token_shift_grad_custom
                    .into_data()
                    .convert::<f32>(),
                Tolerance::default(),
            );
    }

    #[test]
    #[should_panic(expected = "ShapeMismatch")]
    fn shape_mismatch_panics() {
        let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();

        token_shift_diff_custom(TokenShiftDiffForwardInputs {
            embedded_context: Tensor::<TestBackend, 3>::random(
                [2, 4, 8],
                Distribution::Default,
                &device,
            ),
            embedded_token_shift: Tensor::<TestBackend, 2>::random(
                [3, 7],
                Distribution::Default,
                &device,
            ),
            batch_ids: Tensor::<TestBackend, 1, Int>::from_ints([0, 1], &device),
        });
    }
}
