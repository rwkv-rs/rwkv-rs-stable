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

use crate::kernels::train::{
    layout::assert_linear_readable,
    time_mixer::wkv7::io::{
        Wkv7PretrainForwardInputs,
        Wkv7PretrainForwardPrimitiveInputs,
        Wkv7StatepassForwardInputs,
        Wkv7StatepassForwardOutput,
        Wkv7StatepassForwardPrimitiveInputs,
        Wkv7StatepassForwardPrimitiveOutput,
        Wkv7StatetuneForwardInputs,
        Wkv7StatetuneForwardPrimitiveInputs,
    },
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
        assert_linear_readable("receptance", &inputs.receptance);
        assert_linear_readable("weight_decay", &inputs.weight_decay);
        assert_linear_readable("replacement_key", &inputs.replacement_key);
        assert_linear_readable("value", &inputs.value);
        assert_linear_readable("removal_key_normalized", &inputs.removal_key_normalized);
        assert_linear_readable("replacement", &inputs.replacement);

        forward::fused_wkv7_pretrain::<R, F, I, BT>(inputs)
    }

    fn fused_wkv7_statetune(
        inputs: Wkv7StatetuneForwardPrimitiveInputs<Self>,
    ) -> FloatTensor<Self> {
        assert_linear_readable("initial_state", &inputs.initial_state);

        forward::fused_wkv7_statetune::<R, F, I, BT>(inputs)
    }

    fn fused_wkv7_statepass(
        inputs: Wkv7StatepassForwardPrimitiveInputs<Self>,
    ) -> Wkv7StatepassForwardPrimitiveOutput<Self> {
        assert_linear_readable("initial_state", &inputs.initial_state);

        forward::fused_wkv7_statepass::<R, F, I, BT>(inputs)
    }
}

#[cfg(feature = "fusion")]
mod fusion_impl {
    use burn::tensor::Element;
    use burn_fusion::{
        Fusion,
        FusionBackend,
        FusionRuntime,
        stream::{Operation, OperationStreams},
    };
    use burn_ir::{CustomOpIr, HandleContainer, OperationIr, TensorIr};

    use super::*;

    impl<B: FusionBackend + Wkv7Backend> Wkv7Backend for Fusion<B> {
        fn fused_wkv7_pretrain(
            inputs: Wkv7PretrainForwardPrimitiveInputs<Self>,
        ) -> FloatTensor<Self> {
            let Wkv7PretrainForwardPrimitiveInputs {
                receptance,
                weight_decay,
                replacement_key,
                value,
                removal_key_normalized,
                replacement,
                chunk_len,
            } = inputs;
            let client = receptance.client.clone();
            let output_shape = receptance.shape.clone();

            #[derive(Clone, Debug)]
            struct Wkv7PretrainOp<B1> {
                desc: CustomOpIr,
                chunk_len: usize,
                _backend: core::marker::PhantomData<B1>,
            }

            impl<B1: FusionBackend + Wkv7Backend> Operation<B1::FusionRuntime> for Wkv7PretrainOp<B1> {
                fn execute(
                    &self,
                    handles: &mut HandleContainer<
                        <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                    >,
                ) {
                    let (
                        [
                            receptance,
                            weight_decay,
                            replacement_key,
                            value,
                            removal_key_normalized,
                            replacement,
                        ],
                        [output_out],
                    ) = self.desc.as_fixed();

                    let output = B1::fused_wkv7_pretrain(Wkv7PretrainForwardPrimitiveInputs {
                        receptance: handles.get_float_tensor::<B1>(receptance),
                        weight_decay: handles.get_float_tensor::<B1>(weight_decay),
                        replacement_key: handles.get_float_tensor::<B1>(replacement_key),
                        value: handles.get_float_tensor::<B1>(value),
                        removal_key_normalized: handles
                            .get_float_tensor::<B1>(removal_key_normalized),
                        replacement: handles.get_float_tensor::<B1>(replacement),
                        chunk_len: self.chunk_len,
                    });

                    handles.register_float_tensor::<B1>(&output_out.id, output);
                }
            }

            let mut streams = OperationStreams::default();
            streams.tensor(&receptance);
            streams.tensor(&weight_decay);
            streams.tensor(&replacement_key);
            streams.tensor(&value);
            streams.tensor(&removal_key_normalized);
            streams.tensor(&replacement);

            let output_desc = [TensorIr::uninit(
                client.create_empty_handle(),
                output_shape,
                B::FloatElem::dtype(),
            )];
            let desc = CustomOpIr::new(
                "fused_wkv7_pretrain",
                &[
                    receptance.into_ir(),
                    weight_decay.into_ir(),
                    replacement_key.into_ir(),
                    value.into_ir(),
                    removal_key_normalized.into_ir(),
                    replacement.into_ir(),
                ],
                &output_desc,
            );
            let op = Wkv7PretrainOp::<B> {
                desc,
                chunk_len,
                _backend: core::marker::PhantomData,
            };

            client
                .register(streams, OperationIr::Custom(op.desc.clone()), op)
                .pop()
                .expect("missing fused_wkv7_pretrain output")
        }

        fn fused_wkv7_statetune(
            inputs: Wkv7StatetuneForwardPrimitiveInputs<Self>,
        ) -> FloatTensor<Self> {
            let Wkv7StatetuneForwardPrimitiveInputs {
                initial_state,
                sequence,
            } = inputs;
            let Wkv7PretrainForwardPrimitiveInputs {
                receptance,
                weight_decay,
                replacement_key,
                value,
                removal_key_normalized,
                replacement,
                chunk_len,
            } = sequence;
            let client = receptance.client.clone();
            let output_shape = receptance.shape.clone();

            #[derive(Clone, Debug)]
            struct Wkv7StatetuneOp<B1> {
                desc: CustomOpIr,
                chunk_len: usize,
                _backend: core::marker::PhantomData<B1>,
            }

            impl<B1: FusionBackend + Wkv7Backend> Operation<B1::FusionRuntime> for Wkv7StatetuneOp<B1> {
                fn execute(
                    &self,
                    handles: &mut HandleContainer<
                        <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                    >,
                ) {
                    let (
                        [
                            initial_state,
                            receptance,
                            weight_decay,
                            replacement_key,
                            value,
                            removal_key_normalized,
                            replacement,
                        ],
                        [output_out],
                    ) = self.desc.as_fixed();

                    let output = B1::fused_wkv7_statetune(Wkv7StatetuneForwardPrimitiveInputs {
                        initial_state: handles.get_float_tensor::<B1>(initial_state),
                        sequence: Wkv7PretrainForwardPrimitiveInputs {
                            receptance: handles.get_float_tensor::<B1>(receptance),
                            weight_decay: handles.get_float_tensor::<B1>(weight_decay),
                            replacement_key: handles.get_float_tensor::<B1>(replacement_key),
                            value: handles.get_float_tensor::<B1>(value),
                            removal_key_normalized: handles
                                .get_float_tensor::<B1>(removal_key_normalized),
                            replacement: handles.get_float_tensor::<B1>(replacement),
                            chunk_len: self.chunk_len,
                        },
                    });

                    handles.register_float_tensor::<B1>(&output_out.id, output);
                }
            }

            let mut streams = OperationStreams::default();
            streams.tensor(&initial_state);
            streams.tensor(&receptance);
            streams.tensor(&weight_decay);
            streams.tensor(&replacement_key);
            streams.tensor(&value);
            streams.tensor(&removal_key_normalized);
            streams.tensor(&replacement);

            let output_desc = [TensorIr::uninit(
                client.create_empty_handle(),
                output_shape,
                B::FloatElem::dtype(),
            )];
            let desc = CustomOpIr::new(
                "fused_wkv7_statetune",
                &[
                    initial_state.into_ir(),
                    receptance.into_ir(),
                    weight_decay.into_ir(),
                    replacement_key.into_ir(),
                    value.into_ir(),
                    removal_key_normalized.into_ir(),
                    replacement.into_ir(),
                ],
                &output_desc,
            );
            let op = Wkv7StatetuneOp::<B> {
                desc,
                chunk_len,
                _backend: core::marker::PhantomData,
            };

            client
                .register(streams, OperationIr::Custom(op.desc.clone()), op)
                .pop()
                .expect("missing fused_wkv7_statetune output")
        }

        fn fused_wkv7_statepass(
            inputs: Wkv7StatepassForwardPrimitiveInputs<Self>,
        ) -> Wkv7StatepassForwardPrimitiveOutput<Self> {
            let Wkv7StatepassForwardPrimitiveInputs {
                initial_state,
                sequence,
            } = inputs;
            let Wkv7PretrainForwardPrimitiveInputs {
                receptance,
                weight_decay,
                replacement_key,
                value,
                removal_key_normalized,
                replacement,
                chunk_len,
            } = sequence;
            let client = receptance.client.clone();
            let output_shape = receptance.shape.clone();
            let next_state_shape = initial_state.shape.clone();

            #[derive(Clone, Debug)]
            struct Wkv7StatepassOp<B1> {
                desc: CustomOpIr,
                chunk_len: usize,
                _backend: core::marker::PhantomData<B1>,
            }

            impl<B1: FusionBackend + Wkv7Backend> Operation<B1::FusionRuntime> for Wkv7StatepassOp<B1> {
                fn execute(
                    &self,
                    handles: &mut HandleContainer<
                        <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                    >,
                ) {
                    let (
                        [
                            initial_state,
                            receptance,
                            weight_decay,
                            replacement_key,
                            value,
                            removal_key_normalized,
                            replacement,
                        ],
                        [output_out, next_state_out],
                    ) = self.desc.as_fixed();

                    let output = B1::fused_wkv7_statepass(Wkv7StatepassForwardPrimitiveInputs {
                        initial_state: handles.get_float_tensor::<B1>(initial_state),
                        sequence: Wkv7PretrainForwardPrimitiveInputs {
                            receptance: handles.get_float_tensor::<B1>(receptance),
                            weight_decay: handles.get_float_tensor::<B1>(weight_decay),
                            replacement_key: handles.get_float_tensor::<B1>(replacement_key),
                            value: handles.get_float_tensor::<B1>(value),
                            removal_key_normalized: handles
                                .get_float_tensor::<B1>(removal_key_normalized),
                            replacement: handles.get_float_tensor::<B1>(replacement),
                            chunk_len: self.chunk_len,
                        },
                    });

                    handles.register_float_tensor::<B1>(&output_out.id, output.output);
                    handles.register_float_tensor::<B1>(&next_state_out.id, output.next_state);
                }
            }

            let mut streams = OperationStreams::default();
            streams.tensor(&initial_state);
            streams.tensor(&receptance);
            streams.tensor(&weight_decay);
            streams.tensor(&replacement_key);
            streams.tensor(&value);
            streams.tensor(&removal_key_normalized);
            streams.tensor(&replacement);

            let output_desc = [
                TensorIr::uninit(
                    client.create_empty_handle(),
                    output_shape,
                    B::FloatElem::dtype(),
                ),
                TensorIr::uninit(
                    client.create_empty_handle(),
                    next_state_shape,
                    B::FloatElem::dtype(),
                ),
            ];
            let desc = CustomOpIr::new(
                "fused_wkv7_statepass",
                &[
                    initial_state.into_ir(),
                    receptance.into_ir(),
                    weight_decay.into_ir(),
                    replacement_key.into_ir(),
                    value.into_ir(),
                    removal_key_normalized.into_ir(),
                    replacement.into_ir(),
                ],
                &output_desc,
            );
            let op = Wkv7StatepassOp::<B> {
                desc,
                chunk_len,
                _backend: core::marker::PhantomData,
            };

            let mut outputs = client.register(streams, OperationIr::Custom(op.desc.clone()), op);
            let next_state = outputs
                .pop()
                .expect("missing fused_wkv7_statepass next_state");
            let output = outputs.pop().expect("missing fused_wkv7_statepass output");

            Wkv7StatepassForwardPrimitiveOutput { output, next_state }
        }
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
