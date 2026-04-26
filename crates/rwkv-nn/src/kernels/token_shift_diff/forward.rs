use burn::tensor::DType;
use burn_cubecl::{
    CubeBackend,
    CubeElement,
    CubeRuntime,
    FloatElement,
    IntElement,
    cubecl::{calculate_cube_count_elemwise, prelude::*},
    kernel::{cast, into_contiguous},
    ops::numeric::empty_device,
};

use crate::kernels::token_shift_diff::{
    io::{TokenShiftDiffForwardPrimitiveInputs, TokenShiftDiffForwardPrimitiveOutput},
    kernel::{
        TokenShiftDiffInputsLaunch,
        TokenShiftDiffOutputsLaunch,
        fused_token_shift_diff_kernel,
    },
};

pub(crate) fn fused_token_shift_diff<
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: burn_cubecl::element::BoolElement,
>(
    inputs: TokenShiftDiffForwardPrimitiveInputs<CubeBackend<R, F, I, BT>>,
) -> TokenShiftDiffForwardPrimitiveOutput<CubeBackend<R, F, I, BT>> {
    let TokenShiftDiffForwardPrimitiveInputs {
        embedded_context,
        embedded_token_shift,
        batch_ids,
    } = inputs;
    let embedded_context = into_contiguous(embedded_context);
    let embedded_token_shift = into_contiguous(embedded_token_shift);
    let batch_ids = cast::<R>(into_contiguous(batch_ids), DType::U32);

    let client = embedded_context.client.clone();
    let device = embedded_context.device.clone();
    let context_shape = embedded_context.meta.shape().clone();
    let state_shape = embedded_token_shift.meta.shape().clone();
    let token_shifted_diff = empty_device::<R, F>(client.clone(), device, context_shape.clone());
    let active_feature_count = context_shape[0] * context_shape[2];

    if context_shape[1] == 0 || active_feature_count == 0 {
        return TokenShiftDiffForwardPrimitiveOutput {
            token_shifted_diff,
            next_token_shift: embedded_token_shift,
        };
    }

    let cube_dim = CubeDim::new(&client, active_feature_count);
    let cube_count = calculate_cube_count_elemwise(&client, active_feature_count, cube_dim);

    // Each worker owns one `(batch, embedded_dim)` lane, streams all context positions in order,
    // and writes the selected state row once after the final token.
    fused_token_shift_diff_kernel::launch::<F, R>(
        &client,
        cube_count,
        cube_dim,
        TokenShiftDiffInputsLaunch::new(
            embedded_context.into_tensor_arg(),
            batch_ids.into_tensor_arg(),
        ),
        TokenShiftDiffOutputsLaunch::new(
            token_shifted_diff.clone().into_tensor_arg(),
            embedded_token_shift.clone().into_tensor_arg(),
        ),
    );

    debug_assert_eq!(state_shape[1], context_shape[2]);

    TokenShiftDiffForwardPrimitiveOutput {
        token_shifted_diff,
        next_token_shift: embedded_token_shift,
    }
}

#[cfg(feature = "fusion")]
mod fusion_impl {
    use burn::tensor::{Element, Shape};
    use burn_fusion::{
        Fusion,
        FusionBackend,
        FusionRuntime,
        stream::{Operation, OperationStreams},
    };
    use burn_ir::{CustomOpIr, HandleContainer, OperationIr, TensorIr};

    use super::*;
    use crate::kernels::token_shift_diff::TokenShiftDiffBackend;

    impl<B: FusionBackend + TokenShiftDiffBackend> TokenShiftDiffBackend for Fusion<B> {
        fn fused_token_shift_diff(
            inputs: TokenShiftDiffForwardPrimitiveInputs<Self>,
        ) -> TokenShiftDiffForwardPrimitiveOutput<Self> {
            let TokenShiftDiffForwardPrimitiveInputs {
                embedded_context,
                embedded_token_shift,
                batch_ids,
            } = inputs;
            let client = embedded_context.client.clone();
            let [batch_size, context_len, embedded_dim] = embedded_context.shape.dims();
            let [full_batch_size, _] = embedded_token_shift.shape.dims();

            #[derive(Clone, Debug)]
            struct TokenShiftDiffOp<B1> {
                desc: CustomOpIr,
                _backend: core::marker::PhantomData<B1>,
            }

            impl<B1: FusionBackend + TokenShiftDiffBackend> Operation<B1::FusionRuntime>
                for TokenShiftDiffOp<B1>
            {
                fn execute(
                    &self,
                    handles: &mut HandleContainer<
                        <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                    >,
                ) {
                    let (
                        [embedded_context, embedded_token_shift, batch_ids],
                        [token_shifted_diff_out, next_token_shift_out],
                    ) = self.desc.as_fixed();

                    let output = B1::fused_token_shift_diff(TokenShiftDiffForwardPrimitiveInputs {
                        embedded_context: handles.get_float_tensor::<B1>(embedded_context),
                        embedded_token_shift: handles.get_float_tensor::<B1>(embedded_token_shift),
                        batch_ids: handles.get_int_tensor::<B1>(batch_ids),
                    });

                    handles.register_float_tensor::<B1>(
                        &token_shifted_diff_out.id,
                        output.token_shifted_diff,
                    );
                    handles.register_float_tensor::<B1>(
                        &next_token_shift_out.id,
                        output.next_token_shift,
                    );
                }
            }

            let mut streams = OperationStreams::default();
            streams.tensor(&embedded_context);
            streams.tensor(&embedded_token_shift);
            streams.tensor(&batch_ids);

            let output_desc = [
                TensorIr::uninit(
                    client.create_empty_handle(),
                    Shape::new([batch_size, context_len, embedded_dim]),
                    B::FloatElem::dtype(),
                ),
                TensorIr::uninit(
                    client.create_empty_handle(),
                    Shape::new([full_batch_size, embedded_dim]),
                    B::FloatElem::dtype(),
                ),
            ];

            let desc = CustomOpIr::new(
                "fused_token_shift_diff",
                &[
                    embedded_context.into_ir(),
                    embedded_token_shift.into_ir(),
                    batch_ids.into_ir(),
                ],
                &output_desc,
            );

            let op = TokenShiftDiffOp::<B> {
                desc,
                _backend: core::marker::PhantomData,
            };

            let mut outputs = client.register(streams, OperationIr::Custom(op.desc.clone()), op);
            let next_token_shift = outputs.pop().expect("missing next_token_shift output");
            let token_shifted_diff = outputs.pop().expect("missing token_shifted_diff output");

            TokenShiftDiffForwardPrimitiveOutput {
                token_shifted_diff,
                next_token_shift,
            }
        }
    }
}
