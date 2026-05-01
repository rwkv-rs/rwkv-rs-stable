use burn::{
    backend::autodiff::{
        Autodiff,
        checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
        grads::Gradients,
        ops::{Backward, Ops, OpsKind},
    },
    prelude::Int,
    tensor::{IndexingUpdateOp, Tensor, TensorMetadata, TensorPrimitive},
};

use crate::kernels::template::token_shift_diff::{
    TokenShiftDiffBackend,
    io::{
        TokenShiftDiffForwardInputs,
        TokenShiftDiffForwardPrimitiveInputs,
        TokenShiftDiffForwardPrimitiveOutput,
    },
    token_shift_diff_reference,
};

impl<B, C> TokenShiftDiffBackend for Autodiff<B, C>
where
    B: TokenShiftDiffBackend,
    C: CheckpointStrategy,
{
    fn fused_token_shift_diff(
        inputs: TokenShiftDiffForwardPrimitiveInputs<Self>,
    ) -> TokenShiftDiffForwardPrimitiveOutput<Self> {
        #[derive(Debug)]
        struct TokenShiftDiffBackward;

        impl<B: TokenShiftDiffBackend> Backward<B, 2> for TokenShiftDiffBackward {
            type State = (burn::tensor::ops::IntTensor<B>, usize);

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let [node_embedded_context, node_embedded_token_shift] = ops.parents;
                let output_grad = grads.consume::<B>(&ops.node);
                let (batch_ids, full_batch_size) = ops.state;

                let output_grad =
                    Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(output_grad));
                let batch_ids = Tensor::<B, 1, Int>::new(batch_ids);

                let [batch_size, context_len, embedded_dim] = output_grad.dims();
                let device = output_grad.device();

                let shifted_next_grad = if context_len > 1 {
                    Tensor::cat(
                        vec![
                            output_grad.clone().slice([
                                0..batch_size,
                                1..context_len,
                                0..embedded_dim,
                            ]),
                            Tensor::zeros([batch_size, 1, embedded_dim], &device),
                        ],
                        1,
                    )
                } else {
                    Tensor::zeros([batch_size, context_len, embedded_dim], &device)
                };
                let embedded_context_grad = shifted_next_grad - output_grad.clone();

                let active_token_shift_grad = if context_len == 0 {
                    Tensor::zeros([batch_size, embedded_dim], &device)
                } else {
                    output_grad
                        .slice([0..batch_size, 0..1, 0..embedded_dim])
                        .squeeze_dim(1)
                };
                let embedded_token_shift_grad =
                    Tensor::<B, 2>::zeros([full_batch_size, embedded_dim], &device).select_assign(
                        0,
                        batch_ids,
                        active_token_shift_grad,
                        IndexingUpdateOp::Add,
                    );

                if let Some(node) = node_embedded_context {
                    grads.register::<B>(node.id, embedded_context_grad.into_primitive().tensor());
                }
                if let Some(node) = node_embedded_token_shift {
                    grads.register::<B>(
                        node.id,
                        embedded_token_shift_grad.into_primitive().tensor(),
                    );
                }
            }
        }

        let TokenShiftDiffForwardPrimitiveInputs {
            embedded_context,
            embedded_token_shift,
            batch_ids,
        } = inputs;

        let embedded_context_tensor =
            Tensor::<Self, 3>::from_primitive(TensorPrimitive::Float(embedded_context.clone()));
        let embedded_token_shift_tensor =
            Tensor::<Self, 2>::from_primitive(TensorPrimitive::Float(embedded_token_shift.clone()));
        let batch_ids_tensor = Tensor::<Self, 1, Int>::new(batch_ids.clone());
        let context_len = embedded_context.shape()[1];
        let full_batch_size = embedded_token_shift.shape()[0];

        let next_token_shift = token_shift_diff_reference(TokenShiftDiffForwardInputs {
            embedded_context: embedded_context_tensor.clone(),
            embedded_token_shift: embedded_token_shift_tensor,
            batch_ids: batch_ids_tensor,
        })
        .next_token_shift;

        let token_shifted_diff = if context_len == 0 {
            embedded_context
        } else {
            match TokenShiftDiffBackward
                .prepare::<C>([
                    embedded_context.node.clone(),
                    embedded_token_shift.node.clone(),
                ])
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(prep) => {
                    let output = B::fused_token_shift_diff(TokenShiftDiffForwardPrimitiveInputs {
                        embedded_context: embedded_context.primitive.clone(),
                        embedded_token_shift: embedded_token_shift.primitive.clone(),
                        batch_ids: batch_ids.clone(),
                    });
                    prep.finish((batch_ids, full_batch_size), output.token_shifted_diff)
                }
                OpsKind::UnTracked(prep) => prep.finish(
                    B::fused_token_shift_diff(TokenShiftDiffForwardPrimitiveInputs {
                        embedded_context: embedded_context.primitive,
                        embedded_token_shift: embedded_token_shift.primitive,
                        batch_ids,
                    })
                    .token_shifted_diff,
                ),
            }
        };

        TokenShiftDiffForwardPrimitiveOutput {
            token_shifted_diff,
            next_token_shift: next_token_shift.into_primitive().tensor(),
        }
    }
}
