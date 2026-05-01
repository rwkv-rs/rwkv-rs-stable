use burn::tensor::{
    DType,
    Shape,
    ops::{FloatTensor, FloatTensorOps},
};
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
        tune::{AutotuneKey, LocalTuner, Tunable, TunableSet, TuneGroup, anchor, local_tuner},
    },
    element::BoolElement,
    ops::numeric::empty_device,
    tensor::CubeTensor,
};
use serde::{Deserialize, Serialize};

use crate::kernels::train::channel_mixer::{
    io::ChannelMixerForwardPrimitiveInputs,
    kernel::{channel_mixer_mix_forward_kernel, channel_mixer_relu_square_forward_kernel},
};

const LINE_SIZE_CANDIDATES: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
struct ChannelMixerElementwiseAutotuneKey {
    dtype: DType,
    num_elements: usize,
    innermost_dim: usize,
    max_line_size: usize,
}

impl core::fmt::Display for ChannelMixerElementwiseAutotuneKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{:?}:{}:{}:{}",
            self.dtype, self.num_elements, self.innermost_dim, self.max_line_size
        )
    }
}

impl AutotuneKey for ChannelMixerElementwiseAutotuneKey {}

pub(crate) fn fused_channel_mixer<
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
>(
    inputs: ChannelMixerForwardPrimitiveInputs<CubeBackend<R, F, I, BT>>,
) -> FloatTensor<CubeBackend<R, F, I, BT>> {
    let ChannelMixerForwardPrimitiveInputs {
        embedded_context,
        key_scale,
        key_weight,
        value_weight,
    } = inputs;
    let key_input = channel_mixer_mix::<R, F>(embedded_context, key_scale);
    let key_input_shape = key_input.meta.shape().clone();
    let [batch_size, context_len, embedded_dim] = key_input_shape.dims();
    let rows = batch_size * context_len;
    let key_input =
        CubeBackend::<R, F, I, BT>::float_reshape(key_input, Shape::new([rows, embedded_dim]));
    let key_projection = CubeBackend::<R, F, I, BT>::float_matmul(
        key_input,
        CubeBackend::<R, F, I, BT>::float_transpose(key_weight),
    );
    let activated_key = channel_mixer_relu_square::<R, F>(key_projection);
    let output = CubeBackend::<R, F, I, BT>::float_matmul(
        activated_key,
        CubeBackend::<R, F, I, BT>::float_transpose(value_weight),
    );

    CubeBackend::<R, F, I, BT>::float_reshape(
        output,
        Shape::new([batch_size, context_len, embedded_dim]),
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
    use crate::kernels::train::channel_mixer::ChannelMixerBackend;

    impl<B: FusionBackend + ChannelMixerBackend> ChannelMixerBackend for Fusion<B> {
        fn fused_channel_mixer(
            inputs: ChannelMixerForwardPrimitiveInputs<Self>,
        ) -> FloatTensor<Self> {
            let ChannelMixerForwardPrimitiveInputs {
                embedded_context,
                key_scale,
                key_weight,
                value_weight,
            } = inputs;
            let client = embedded_context.client.clone();
            let [batch_size, context_len, _] = embedded_context.shape.dims();
            let [embedded_dim, _] = value_weight.shape.dims();

            #[derive(Clone, Debug)]
            struct ChannelMixerOp<B1> {
                desc: CustomOpIr,
                _backend: core::marker::PhantomData<B1>,
            }

            impl<B1: FusionBackend + ChannelMixerBackend> Operation<B1::FusionRuntime> for ChannelMixerOp<B1> {
                fn execute(
                    &self,
                    handles: &mut HandleContainer<
                        <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                    >,
                ) {
                    let ([embedded_context, key_scale, key_weight, value_weight], [output_out]) =
                        self.desc.as_fixed();

                    let output = B1::fused_channel_mixer(ChannelMixerForwardPrimitiveInputs {
                        embedded_context: handles.get_float_tensor::<B1>(embedded_context),
                        key_scale: handles.get_float_tensor::<B1>(key_scale),
                        key_weight: handles.get_float_tensor::<B1>(key_weight),
                        value_weight: handles.get_float_tensor::<B1>(value_weight),
                    });

                    handles.register_float_tensor::<B1>(&output_out.id, output);
                }
            }

            let mut streams = OperationStreams::default();
            streams.tensor(&embedded_context);
            streams.tensor(&key_scale);
            streams.tensor(&key_weight);
            streams.tensor(&value_weight);

            let output_desc = [TensorIr::uninit(
                client.create_empty_handle(),
                Shape::new([batch_size, context_len, embedded_dim]),
                B::FloatElem::dtype(),
            )];

            let desc = CustomOpIr::new(
                "fused_channel_mixer",
                &[
                    embedded_context.into_ir(),
                    key_scale.into_ir(),
                    key_weight.into_ir(),
                    value_weight.into_ir(),
                ],
                &output_desc,
            );

            let op = ChannelMixerOp::<B> {
                desc,
                _backend: core::marker::PhantomData,
            };

            client
                .register(streams, OperationIr::Custom(op.desc.clone()), op)
                .pop()
                .expect("missing fused_channel_mixer output")
        }
    }
}

pub(super) fn channel_mixer_mix<R, F>(
    embedded_context: CubeTensor<R>,
    key_scale: CubeTensor<R>,
) -> CubeTensor<R>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
{
    let client = embedded_context.client.clone();

    let key = |(embedded_context, key_scale): &(CubeTensor<R>, CubeTensor<R>)| {
        let shape = embedded_context.meta.shape();

        ChannelMixerElementwiseAutotuneKey {
            dtype: embedded_context.dtype,
            num_elements: anchor(shape.num_elements(), None, Some(1), None),
            innermost_dim: shape[shape.num_dims() - 1],
            max_line_size: max_line_size_many(&[embedded_context, key_scale], shape.num_dims() - 1),
        }
    };

    let input_gen =
        |_key: &ChannelMixerElementwiseAutotuneKey,
         (embedded_context, key_scale): &(CubeTensor<R>, CubeTensor<R>)| {
            (embedded_context.copy(), key_scale.copy())
        };

    static TUNER: LocalTuner<ChannelMixerElementwiseAutotuneKey, CubeTuneId> =
        local_tuner!("channel-mixer-mix-forward");

    let tunables = TUNER.init(move || {
        let line_size_group =
            TuneGroup::<ChannelMixerElementwiseAutotuneKey>::new("line_size", |_| 1);
        let mut set = TunableSet::new(key, input_gen);

        for line_size in LINE_SIZE_CANDIDATES {
            set = set.with(
                Tunable::new(
                    &format!("line_size_{line_size}"),
                    move |(embedded_context, key_scale)| {
                        Ok::<_, String>(channel_mixer_mix_with_vector::<R, F>(
                            embedded_context,
                            key_scale,
                            line_size,
                        ))
                    },
                )
                .group(&line_size_group, move |key| {
                    if line_size <= key.max_line_size && key.innermost_dim.is_multiple_of(line_size)
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
        (embedded_context, key_scale),
    )
}

fn channel_mixer_mix_with_vector<R, F>(
    embedded_context: CubeTensor<R>,
    key_scale: CubeTensor<R>,
    vector_size: usize,
) -> CubeTensor<R>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
{
    let shape = embedded_context.meta.shape().clone();
    let client = embedded_context.client.clone();
    let key_input = empty_device::<R, F>(
        client.clone(),
        embedded_context.device.clone(),
        shape.clone(),
    );

    if shape.num_elements() == 0 {
        return key_input;
    }

    let working_units = shape.num_elements() / vector_size;
    let cube_dim = CubeDim::new(&client, working_units);
    let cube_count = calculate_cube_count_elemwise(&client, working_units, cube_dim);
    let address_type = max_address_type(&[&embedded_context, &key_scale, &key_input]);

    // The mix stage is independent per `[batch_size, context_len, embedded_dim]` element after
    // reading the previous time position. Each work unit handles contiguous embedded-dimension
    // lanes. Live state is current token, previous token, scale, and output. The first time
    // position uses zero, which prevents cross-batch reads at sequence boundaries.
    // SAFETY: The public contract checks dtype/device/shape/contiguity. Autotune only keeps vector
    // sizes that divide `embedded_dim`, so linear vector views cover the logical element range.
    unsafe {
        channel_mixer_mix_forward_kernel::launch_unchecked::<F, R>(
            &client,
            cube_count,
            cube_dim,
            address_type,
            vector_size,
            embedded_context.into_linear_view_like(&key_input),
            key_scale.into_linear_view(),
            shape[1],
            key_input.clone().into_linear_view(),
        );
    }

    key_input
}

pub(super) fn channel_mixer_relu_square<R, F>(pre_activation: CubeTensor<R>) -> CubeTensor<R>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
{
    let client = pre_activation.client.clone();

    let key = |pre_activation: &CubeTensor<R>| {
        let shape = pre_activation.meta.shape();

        ChannelMixerElementwiseAutotuneKey {
            dtype: pre_activation.dtype,
            num_elements: anchor(shape.num_elements(), None, Some(1), None),
            innermost_dim: shape[shape.num_dims() - 1],
            max_line_size: max_line_size_many(&[pre_activation], shape.num_dims() - 1),
        }
    };

    let input_gen = |_key: &ChannelMixerElementwiseAutotuneKey, pre_activation: &CubeTensor<R>| {
        pre_activation.copy()
    };

    static TUNER: LocalTuner<ChannelMixerElementwiseAutotuneKey, CubeTuneId> =
        local_tuner!("channel-mixer-relu-square-forward");

    let tunables = TUNER.init(move || {
        let line_size_group =
            TuneGroup::<ChannelMixerElementwiseAutotuneKey>::new("line_size", |_| 1);
        let mut set = TunableSet::new(key, input_gen);

        for line_size in LINE_SIZE_CANDIDATES {
            set = set.with(
                Tunable::new(&format!("line_size_{line_size}"), move |pre_activation| {
                    Ok::<_, String>(channel_mixer_relu_square_with_vector::<R, F>(
                        pre_activation,
                        line_size,
                    ))
                })
                .group(&line_size_group, move |key| {
                    if line_size <= key.max_line_size && key.innermost_dim.is_multiple_of(line_size)
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
        &CubeTuneId::new(&pre_activation.client, &pre_activation.device),
        &client,
        tunables,
        pre_activation,
    )
}

fn channel_mixer_relu_square_with_vector<R, F>(
    pre_activation: CubeTensor<R>,
    vector_size: usize,
) -> CubeTensor<R>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
{
    let shape = pre_activation.meta.shape().clone();
    let client = pre_activation.client.clone();
    let activated_key =
        empty_device::<R, F>(client.clone(), pre_activation.device.clone(), shape.clone());

    if shape.num_elements() == 0 {
        return activated_key;
    }

    let working_units = shape.num_elements() / vector_size;
    let cube_dim = CubeDim::new(&client, working_units);
    let cube_count = calculate_cube_count_elemwise(&client, working_units, cube_dim);
    let address_type = max_address_type(&[&pre_activation, &activated_key]);

    // ReLU-square has one independent output per key projection element. Vectorizing along the
    // expanded dimension keeps the projected activation reads and writes contiguous. Live state is
    // the projection value, zero clamp, and squared output.
    // SAFETY: The input is contiguous and the tuner only keeps vector sizes that divide the
    // innermost expanded dimension, so vector views cover the logical element range.
    unsafe {
        channel_mixer_relu_square_forward_kernel::launch_unchecked::<F, R>(
            &client,
            cube_count,
            cube_dim,
            address_type,
            vector_size,
            pre_activation.into_linear_view_like(&activated_key),
            activated_key.clone().into_linear_view(),
        );
    }

    activated_key
}

fn max_line_size_many<R: CubeRuntime>(tensors: &[&CubeTensor<R>], axis: usize) -> usize {
    tensors
        .iter()
        .map(|tensor| {
            let tensor_axis = axis.min(tensor.meta.shape().num_dims().saturating_sub(1));
            tensor_vector_size_parallel(
                tensor.client.io_optimized_vector_sizes(tensor.dtype.size()),
                tensor.meta.shape(),
                tensor.meta.strides(),
                tensor_axis,
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
