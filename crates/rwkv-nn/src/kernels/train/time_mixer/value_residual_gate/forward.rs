use burn::tensor::{DType, ops::FloatTensor};
use burn_cubecl::{
    CubeBackend,
    CubeElement,
    CubeRuntime,
    CubeTuneId,
    FloatElement,
    IntElement,
    cubecl::{
        calculate_cube_count_elemwise,
        prelude::*,
        tensor_vector_size_parallel,
        tune::{
            AsFunctionTunable,
            AutotuneKey,
            LocalTuner,
            Tunable,
            TunableSet,
            TuneGroup,
            anchor,
            local_tuner,
        },
    },
    element::BoolElement,
    ops::numeric::empty_device,
    tensor::CubeTensor,
};
use serde::{Deserialize, Serialize};

use crate::kernels::train::time_mixer::value_residual_gate::{
    io::ValueResidualGateForwardPrimitiveInputs,
    kernel::value_residual_gate_forward_kernel,
};

const LINE_SIZE_CANDIDATES: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
struct ValueResidualGateForwardAutotuneKey {
    dtype: DType,
    num_elements: usize,
    embedded_dim: usize,
    max_line_size: usize,
}

impl core::fmt::Display for ValueResidualGateForwardAutotuneKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{:?}:{}:{}:{}",
            self.dtype, self.num_elements, self.embedded_dim, self.max_line_size
        )
    }
}

impl AutotuneKey for ValueResidualGateForwardAutotuneKey {}

pub(crate) fn fused_value_residual_gate<
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
>(
    inputs: ValueResidualGateForwardPrimitiveInputs<CubeBackend<R, F, I, BT>>,
) -> FloatTensor<CubeBackend<R, F, I, BT>> {
    let ValueResidualGateForwardPrimitiveInputs {
        value,
        value_from_first_cell,
        gate_base,
        gate_input,
    } = inputs;
    let client = value.client.clone();

    let key = |value: &CubeTensor<R>,
               value_from_first_cell: &CubeTensor<R>,
               gate_base: &CubeTensor<R>,
               gate_input: &CubeTensor<R>| {
        let shape = value.meta.shape();

        ValueResidualGateForwardAutotuneKey {
            dtype: value.dtype,
            num_elements: anchor(shape.num_elements(), None, Some(1), None),
            embedded_dim: shape[2],
            max_line_size: max_line_size_many(
                &[value, value_from_first_cell, gate_base, gate_input],
                shape.num_dims() - 1,
            ),
        }
    };

    let input_gen = |_key: &ValueResidualGateForwardAutotuneKey,
                     value: &CubeTensor<R>,
                     value_from_first_cell: &CubeTensor<R>,
                     gate_base: &CubeTensor<R>,
                     gate_input: &CubeTensor<R>| {
        (
            value.copy(),
            value_from_first_cell.copy(),
            gate_base.copy(),
            gate_input.copy(),
        )
    };

    static TUNER: LocalTuner<ValueResidualGateForwardAutotuneKey, CubeTuneId> =
        local_tuner!("value-residual-gate-forward");

    let tunables = TUNER.init(move || {
        let line_size_group =
            TuneGroup::<ValueResidualGateForwardAutotuneKey>::new("line_size", |_| 1);
        let mut set = TunableSet::new(key, input_gen);

        for line_size in LINE_SIZE_CANDIDATES {
            set = set.with(
                Tunable::new(
                    format!("line_size_{line_size}"),
                    (move |value: CubeTensor<R>,
                           value_from_first_cell: CubeTensor<R>,
                           gate_base: CubeTensor<R>,
                           gate_input: CubeTensor<R>| {
                        value_residual_gate::<R, F>(
                            value,
                            value_from_first_cell,
                            gate_base,
                            gate_input,
                            line_size,
                        )
                    })
                    .ok(),
                )
                .group(&line_size_group, move |key| {
                    if line_size <= key.max_line_size && key.embedded_dim.is_multiple_of(line_size)
                    {
                        1
                    } else {
                        -1
                    }
                }),
            );
        }

        set
    });

    TUNER.execute(
        &CubeTuneId::new(&value.client, &value.device),
        &client,
        tunables,
        (value, value_from_first_cell, gate_base, gate_input),
    )
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
    use crate::kernels::train::time_mixer::value_residual_gate::ValueResidualGateBackend;

    impl<B: FusionBackend + ValueResidualGateBackend> ValueResidualGateBackend for Fusion<B> {
        fn fused_value_residual_gate(
            inputs: ValueResidualGateForwardPrimitiveInputs<Self>,
        ) -> FloatTensor<Self> {
            let ValueResidualGateForwardPrimitiveInputs {
                value,
                value_from_first_cell,
                gate_base,
                gate_input,
            } = inputs;
            let client = value.client.clone();
            let [batch_size, context_len, embedded_dim] = value.shape.dims();

            #[derive(Clone, Debug)]
            struct ValueResidualGateOp<B1> {
                desc: CustomOpIr,
                _backend: core::marker::PhantomData<B1>,
            }

            impl<B1: FusionBackend + ValueResidualGateBackend> Operation<B1::FusionRuntime>
                for ValueResidualGateOp<B1>
            {
                fn execute(
                    &self,
                    handles: &mut HandleContainer<
                        <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                    >,
                ) {
                    let ([value, value_from_first_cell, gate_base, gate_input], [output_out]) =
                        self.desc.as_fixed();

                    let output =
                        B1::fused_value_residual_gate(ValueResidualGateForwardPrimitiveInputs {
                            value: handles.get_float_tensor::<B1>(value),
                            value_from_first_cell: handles
                                .get_float_tensor::<B1>(value_from_first_cell),
                            gate_base: handles.get_float_tensor::<B1>(gate_base),
                            gate_input: handles.get_float_tensor::<B1>(gate_input),
                        });

                    handles.register_float_tensor::<B1>(&output_out.id, output);
                }
            }

            let mut streams = OperationStreams::default();
            streams.tensor(&value);
            streams.tensor(&value_from_first_cell);
            streams.tensor(&gate_base);
            streams.tensor(&gate_input);

            let output_desc = [TensorIr::uninit(
                client.create_empty_handle(),
                Shape::new([batch_size, context_len, embedded_dim]),
                B::FloatElem::dtype(),
            )];

            let desc = CustomOpIr::new(
                "fused_value_residual_gate",
                &[
                    value.into_ir(),
                    value_from_first_cell.into_ir(),
                    gate_base.into_ir(),
                    gate_input.into_ir(),
                ],
                &output_desc,
            );

            let op = ValueResidualGateOp::<B> {
                desc,
                _backend: core::marker::PhantomData,
            };

            client
                .register(streams, OperationIr::Custom(op.desc.clone()), op)
                .pop()
                .expect("missing fused_value_residual_gate output")
        }
    }
}

fn value_residual_gate<R, F>(
    value: CubeTensor<R>,
    value_from_first_cell: CubeTensor<R>,
    gate_base: CubeTensor<R>,
    gate_input: CubeTensor<R>,
    vector_size: usize,
) -> CubeTensor<R>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
{
    let shape = value.meta.shape().clone();
    let client = value.client.clone();
    let output = empty_device::<R, F>(client.clone(), value.device.clone(), shape.clone());

    if shape.num_elements() == 0 {
        return output;
    }

    let working_units = shape.num_elements() / vector_size;
    let cube_dim = CubeDim::new(&client, working_units);
    let cube_count = calculate_cube_count_elemwise(&client, working_units, cube_dim);
    let address_type = max_address_type(&[
        &value,
        &value_from_first_cell,
        &gate_base,
        &gate_input,
        &output,
    ]);

    // The output is independent per `[batch_size, context_len, embedded_dim]` element after
    // broadcasting `gate_base` across the first two axes. Each work unit handles one vector of
    // contiguous embedded-dimension lanes, so the large tensors have coalesced reads and writes
    // when the tuned vector width divides `embedded_dim`. Live state is the current value vector,
    // first-cell value vector, gate base/input vectors, sigmoid temporaries, and output vector.
    // The small gate base is repeatedly indexed by embedded dimension and relies on cache reuse.
    // SAFETY: The public contract checks dtype/device/shape/contiguity. Autotune only keeps vector
    // sizes that divide the embedded dimension, so these linear vector views cover the same logical
    // element range.
    unsafe {
        value_residual_gate_forward_kernel::launch_unchecked::<F, R>(
            &client,
            cube_count,
            cube_dim,
            address_type,
            vector_size,
            value.into_linear_view_like(&output),
            value_from_first_cell.into_linear_view_like(&output),
            gate_base.into_linear_view(),
            gate_input.into_linear_view_like(&output),
            output.clone().into_linear_view(),
        );
    }

    output
}

fn max_line_size_many<R: CubeRuntime>(tensors: &[&CubeTensor<R>], axis: usize) -> usize {
    tensors
        .iter()
        .map(|tensor| {
            tensor_vector_size_parallel(
                tensor.client.io_optimized_vector_sizes(tensor.dtype.size()),
                tensor.meta.shape(),
                tensor.meta.strides(),
                axis,
            )
        })
        .min()
        .unwrap_or(1)
        .max(1)
}

fn max_address_type<R: CubeRuntime>(tensors: &[&CubeTensor<R>]) -> AddressType {
    tensors
        .iter()
        .map(|tensor| tensor.required_address_type())
        .max()
        .unwrap_or_default()
}
