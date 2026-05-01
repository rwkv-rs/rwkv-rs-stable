use burn::tensor::DType;
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
            AutotuneKey,
            AutotuneOutput,
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

use crate::kernels::train::time_mixer::mix6::{
    io::{Mix6ForwardPrimitiveInputs, Mix6ForwardPrimitiveOutput},
    kernel::{Mix6ForwardInputsLaunch, Mix6ForwardOutputsLaunch, mix6_forward_kernel},
};

const LINE_SIZE_CANDIDATES: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
struct Mix6ForwardAutotuneKey {
    dtype: DType,
    num_elements: usize,
    embedded_dim: usize,
    context_len: usize,
    max_line_size: usize,
}

impl core::fmt::Display for Mix6ForwardAutotuneKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{:?}:{}:{}:{}:{}",
            self.dtype, self.num_elements, self.embedded_dim, self.context_len, self.max_line_size
        )
    }
}

impl AutotuneKey for Mix6ForwardAutotuneKey {}

impl<R, F, I, BT> AutotuneOutput for Mix6ForwardPrimitiveOutput<CubeBackend<R, F, I, BT>>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    #[cfg(feature = "autotune-checks")]
    fn check_equivalence(&self, other: Self) {
        AutotuneOutput::check_equivalence(&self.receptance_input, other.receptance_input);
        AutotuneOutput::check_equivalence(&self.weight_decay_input, other.weight_decay_input);
        AutotuneOutput::check_equivalence(&self.key_input, other.key_input);
        AutotuneOutput::check_equivalence(&self.value_input, other.value_input);
        AutotuneOutput::check_equivalence(&self.learning_rate_input, other.learning_rate_input);
        AutotuneOutput::check_equivalence(&self.gate_input, other.gate_input);
    }
}

pub(crate) fn fused_mix6<
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
>(
    inputs: Mix6ForwardPrimitiveInputs<CubeBackend<R, F, I, BT>>,
) -> Mix6ForwardPrimitiveOutput<CubeBackend<R, F, I, BT>> {
    let Mix6ForwardPrimitiveInputs {
        embedded_context,
        receptance_scale,
        weight_decay_scale,
        key_scale,
        value_scale,
        learning_rate_scale,
        gate_scale,
    } = inputs;
    let client = embedded_context.client.clone();

    let key = |(
        embedded_context,
        receptance_scale,
        weight_decay_scale,
        key_scale,
        value_scale,
        learning_rate_scale,
        gate_scale,
    ): &(
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
    )| {
        let shape = embedded_context.meta.shape();

        Mix6ForwardAutotuneKey {
            dtype: embedded_context.dtype,
            num_elements: anchor(shape.num_elements(), None, Some(1), None),
            embedded_dim: shape[2],
            context_len: anchor(shape[1], None, Some(1), None),
            max_line_size: max_line_size_many(
                &[
                    embedded_context,
                    receptance_scale,
                    weight_decay_scale,
                    key_scale,
                    value_scale,
                    learning_rate_scale,
                    gate_scale,
                ],
                shape.num_dims() - 1,
            ),
        }
    };

    let input_gen = |_key: &Mix6ForwardAutotuneKey,
                     (
        embedded_context,
        receptance_scale,
        weight_decay_scale,
        key_scale,
        value_scale,
        learning_rate_scale,
        gate_scale,
    ): &(
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
    )| {
        (
            embedded_context.copy(),
            receptance_scale.copy(),
            weight_decay_scale.copy(),
            key_scale.copy(),
            value_scale.copy(),
            learning_rate_scale.copy(),
            gate_scale.copy(),
        )
    };

    static TUNER: LocalTuner<Mix6ForwardAutotuneKey, CubeTuneId> =
        local_tuner!("rwkv7-time-mixer-mix6-forward");

    let tunables = TUNER.init(move || {
        let line_size_group = TuneGroup::<Mix6ForwardAutotuneKey>::new("line_size", |_| 1);
        let mut set = TunableSet::new(key, input_gen);

        for line_size in LINE_SIZE_CANDIDATES {
            set = set.with(
                Tunable::new(
                    &format!("line_size_{line_size}"),
                    move |(
                        embedded_context,
                        receptance_scale,
                        weight_decay_scale,
                        key_scale,
                        value_scale,
                        learning_rate_scale,
                        gate_scale,
                    )| {
                        Ok::<_, String>(mix6::<R, F, I, BT>(
                            embedded_context,
                            receptance_scale,
                            weight_decay_scale,
                            key_scale,
                            value_scale,
                            learning_rate_scale,
                            gate_scale,
                            line_size,
                        ))
                    },
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
        &CubeTuneId::new(&embedded_context.client, &embedded_context.device),
        &client,
        tunables,
        (
            embedded_context,
            receptance_scale,
            weight_decay_scale,
            key_scale,
            value_scale,
            learning_rate_scale,
            gate_scale,
        ),
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
    use crate::kernels::train::time_mixer::mix6::Mix6Backend;

    impl<B: FusionBackend + Mix6Backend> Mix6Backend for Fusion<B> {
        fn fused_mix6(
            inputs: Mix6ForwardPrimitiveInputs<Self>,
        ) -> Mix6ForwardPrimitiveOutput<Self> {
            let Mix6ForwardPrimitiveInputs {
                embedded_context,
                receptance_scale,
                weight_decay_scale,
                key_scale,
                value_scale,
                learning_rate_scale,
                gate_scale,
            } = inputs;
            let client = embedded_context.client.clone();
            let [batch_size, context_len, embedded_dim] = embedded_context.shape.dims();

            #[derive(Clone, Debug)]
            struct Mix6Op<B1> {
                desc: CustomOpIr,
                _backend: core::marker::PhantomData<B1>,
            }

            impl<B1: FusionBackend + Mix6Backend> Operation<B1::FusionRuntime> for Mix6Op<B1> {
                fn execute(
                    &self,
                    handles: &mut HandleContainer<
                        <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                    >,
                ) {
                    let (
                        [
                            embedded_context,
                            receptance_scale,
                            weight_decay_scale,
                            key_scale,
                            value_scale,
                            learning_rate_scale,
                            gate_scale,
                        ],
                        [
                            receptance_input_out,
                            weight_decay_input_out,
                            key_input_out,
                            value_input_out,
                            learning_rate_input_out,
                            gate_input_out,
                        ],
                    ) = self.desc.as_fixed();

                    let output = B1::fused_mix6(Mix6ForwardPrimitiveInputs {
                        embedded_context: handles.get_float_tensor::<B1>(embedded_context),
                        receptance_scale: handles.get_float_tensor::<B1>(receptance_scale),
                        weight_decay_scale: handles.get_float_tensor::<B1>(weight_decay_scale),
                        key_scale: handles.get_float_tensor::<B1>(key_scale),
                        value_scale: handles.get_float_tensor::<B1>(value_scale),
                        learning_rate_scale: handles.get_float_tensor::<B1>(learning_rate_scale),
                        gate_scale: handles.get_float_tensor::<B1>(gate_scale),
                    });

                    handles.register_float_tensor::<B1>(
                        &receptance_input_out.id,
                        output.receptance_input,
                    );
                    handles.register_float_tensor::<B1>(
                        &weight_decay_input_out.id,
                        output.weight_decay_input,
                    );
                    handles.register_float_tensor::<B1>(&key_input_out.id, output.key_input);
                    handles.register_float_tensor::<B1>(&value_input_out.id, output.value_input);
                    handles.register_float_tensor::<B1>(
                        &learning_rate_input_out.id,
                        output.learning_rate_input,
                    );
                    handles.register_float_tensor::<B1>(&gate_input_out.id, output.gate_input);
                }
            }

            let mut streams = OperationStreams::default();
            streams.tensor(&embedded_context);
            streams.tensor(&receptance_scale);
            streams.tensor(&weight_decay_scale);
            streams.tensor(&key_scale);
            streams.tensor(&value_scale);
            streams.tensor(&learning_rate_scale);
            streams.tensor(&gate_scale);

            let output_desc = [
                TensorIr::uninit(
                    client.create_empty_handle(),
                    Shape::new([batch_size, context_len, embedded_dim]),
                    B::FloatElem::dtype(),
                ),
                TensorIr::uninit(
                    client.create_empty_handle(),
                    Shape::new([batch_size, context_len, embedded_dim]),
                    B::FloatElem::dtype(),
                ),
                TensorIr::uninit(
                    client.create_empty_handle(),
                    Shape::new([batch_size, context_len, embedded_dim]),
                    B::FloatElem::dtype(),
                ),
                TensorIr::uninit(
                    client.create_empty_handle(),
                    Shape::new([batch_size, context_len, embedded_dim]),
                    B::FloatElem::dtype(),
                ),
                TensorIr::uninit(
                    client.create_empty_handle(),
                    Shape::new([batch_size, context_len, embedded_dim]),
                    B::FloatElem::dtype(),
                ),
                TensorIr::uninit(
                    client.create_empty_handle(),
                    Shape::new([batch_size, context_len, embedded_dim]),
                    B::FloatElem::dtype(),
                ),
            ];

            let desc = CustomOpIr::new(
                "fused_mix6",
                &[
                    embedded_context.into_ir(),
                    receptance_scale.into_ir(),
                    weight_decay_scale.into_ir(),
                    key_scale.into_ir(),
                    value_scale.into_ir(),
                    learning_rate_scale.into_ir(),
                    gate_scale.into_ir(),
                ],
                &output_desc,
            );

            let op = Mix6Op::<B> {
                desc,
                _backend: core::marker::PhantomData,
            };

            let mut outputs = client.register(streams, OperationIr::Custom(op.desc.clone()), op);
            let gate_input = outputs.pop().expect("missing gate_input");
            let learning_rate_input = outputs.pop().expect("missing learning_rate_input");
            let value_input = outputs.pop().expect("missing value_input");
            let key_input = outputs.pop().expect("missing key_input");
            let weight_decay_input = outputs.pop().expect("missing weight_decay_input");
            let receptance_input = outputs.pop().expect("missing receptance_input");

            Mix6ForwardPrimitiveOutput {
                receptance_input,
                weight_decay_input,
                key_input,
                value_input,
                learning_rate_input,
                gate_input,
            }
        }
    }
}

fn mix6<R, F, I, BT>(
    embedded_context: CubeTensor<R>,
    receptance_scale: CubeTensor<R>,
    weight_decay_scale: CubeTensor<R>,
    key_scale: CubeTensor<R>,
    value_scale: CubeTensor<R>,
    learning_rate_scale: CubeTensor<R>,
    gate_scale: CubeTensor<R>,
    vector_size: usize,
) -> Mix6ForwardPrimitiveOutput<CubeBackend<R, F, I, BT>>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
{
    let shape = embedded_context.meta.shape().clone();
    let client = embedded_context.client.clone();
    let receptance_input = empty_device::<R, F>(
        client.clone(),
        embedded_context.device.clone(),
        shape.clone(),
    );
    let weight_decay_input = empty_device::<R, F>(
        client.clone(),
        embedded_context.device.clone(),
        shape.clone(),
    );
    let key_input = empty_device::<R, F>(
        client.clone(),
        embedded_context.device.clone(),
        shape.clone(),
    );
    let value_input = empty_device::<R, F>(
        client.clone(),
        embedded_context.device.clone(),
        shape.clone(),
    );
    let learning_rate_input = empty_device::<R, F>(
        client.clone(),
        embedded_context.device.clone(),
        shape.clone(),
    );
    let gate_input = empty_device::<R, F>(
        client.clone(),
        embedded_context.device.clone(),
        shape.clone(),
    );

    if shape.num_elements() == 0 {
        return Mix6ForwardPrimitiveOutput {
            receptance_input,
            weight_decay_input,
            key_input,
            value_input,
            learning_rate_input,
            gate_input,
        };
    }

    let working_units = shape.num_elements() / vector_size;
    let cube_dim = CubeDim::new(&client, working_units);
    let cube_count = calculate_cube_count_elemwise(&client, working_units, cube_dim);
    let address_type = max_address_type(&[
        &embedded_context,
        &receptance_scale,
        &weight_decay_scale,
        &key_scale,
        &value_scale,
        &learning_rate_scale,
        &gate_scale,
        &receptance_input,
        &weight_decay_input,
        &key_input,
        &value_input,
        &learning_rate_input,
        &gate_input,
    ]);

    // Each work unit owns one contiguous embedded-dimension vector for one token. It reads the
    // current token vector and, except at the first time position, the previous token vector from
    // the same batch. The resulting shift difference stays live once and is reused across six
    // branch formulas. Autotune only enables vector sizes that divide `embedded_dim`, so the
    // previous-token offset is an integer number of vector lanes.
    // SAFETY: The public contract checks shapes/dtype/device and primitive dispatch checks
    // contiguity. The tuned vector width divides the embedded axis, and all outputs are allocated
    // with the exact input shape.
    unsafe {
        mix6_forward_kernel::launch_unchecked::<F, R>(
            &client,
            cube_count,
            cube_dim,
            address_type,
            vector_size,
            Mix6ForwardInputsLaunch::new(
                embedded_context.into_linear_view_like(&receptance_input),
                receptance_scale.into_linear_view(),
                weight_decay_scale.into_linear_view(),
                key_scale.into_linear_view(),
                value_scale.into_linear_view(),
                learning_rate_scale.into_linear_view(),
                gate_scale.into_linear_view(),
            ),
            Mix6ForwardOutputsLaunch::new(
                receptance_input.clone().into_linear_view(),
                weight_decay_input.clone().into_linear_view(),
                key_input.clone().into_linear_view(),
                value_input.clone().into_linear_view(),
                learning_rate_input.clone().into_linear_view(),
                gate_input.clone().into_linear_view(),
            ),
            shape[1],
        );
    }

    Mix6ForwardPrimitiveOutput {
        receptance_input,
        weight_decay_input,
        key_input,
        value_input,
        learning_rate_input,
        gate_input,
    }
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
