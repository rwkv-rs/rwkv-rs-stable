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
        tune::{AutotuneKey, LocalTuner, Tunable, TunableSet, TuneGroup, anchor, local_tuner},
    },
    element::BoolElement,
    ops::numeric::empty_device,
    tensor::CubeTensor,
};
use serde::{Deserialize, Serialize};

use crate::kernels::train::{
    layout::CubeHardwareFingerprint,
    time_mixer::weight_decay_transform::{
        io::WeightDecayTransformForwardPrimitiveInputs,
        kernel::{
            weight_decay_transform_forward_kernel,
            weight_decay_transform_forward_pow2_kernel,
        },
    },
};

const LINE_SIZE_CANDIDATES: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
struct WeightDecayTransformForwardAutotuneKey {
    runtime: String,
    dtype: DType,
    num_elements: usize,
    embedded_dim: usize,
    rows: usize,
    hardware: CubeHardwareFingerprint,
    max_line_size: usize,
    is_in_place: bool,
    deterministic: bool,
}

impl core::fmt::Display for WeightDecayTransformForwardAutotuneKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{}:{:?}:n{}:d{}:r{}:{}:line{}:inplace{}:det{}",
            self.runtime,
            self.dtype,
            self.num_elements,
            self.embedded_dim,
            self.rows,
            self.hardware,
            self.max_line_size,
            self.is_in_place,
            self.deterministic
        )
    }
}

impl AutotuneKey for WeightDecayTransformForwardAutotuneKey {}

pub(crate) fn fused_weight_decay_transform<
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
>(
    inputs: WeightDecayTransformForwardPrimitiveInputs<CubeBackend<R, F, I, BT>>,
) -> FloatTensor<CubeBackend<R, F, I, BT>> {
    let WeightDecayTransformForwardPrimitiveInputs {
        weight_decay_base,
        weight_decay_input,
    } = inputs;
    let client = weight_decay_input.client.clone();

    let key = |(weight_decay_base, weight_decay_input): &(CubeTensor<R>, CubeTensor<R>)| {
        let shape = weight_decay_input.meta.shape();
        let embedded_dim = shape[2];
        let hardware = CubeHardwareFingerprint::from_hardware(
            &weight_decay_input.client.properties().hardware,
        );

        WeightDecayTransformForwardAutotuneKey {
            runtime: R::name(&weight_decay_input.client).to_owned(),
            dtype: weight_decay_input.dtype,
            num_elements: anchor(shape.num_elements(), None, Some(1), None),
            embedded_dim,
            rows: anchor(shape.num_elements() / embedded_dim, None, Some(1), None),
            hardware,
            max_line_size: max_line_size_pair(weight_decay_base, weight_decay_input),
            is_in_place: false,
            deterministic: true,
        }
    };

    let input_gen =
        |_key: &WeightDecayTransformForwardAutotuneKey,
         (weight_decay_base, weight_decay_input): &(CubeTensor<R>, CubeTensor<R>)| {
            (weight_decay_base.copy(), weight_decay_input.copy())
        };

    static TUNER: LocalTuner<WeightDecayTransformForwardAutotuneKey, CubeTuneId> =
        local_tuner!("weight-decay-transform-forward");

    let tunables = TUNER.init(move || {
        let line_size_group =
            TuneGroup::<WeightDecayTransformForwardAutotuneKey>::new("line_size", |_| 1);
        let mut set = TunableSet::new(key, input_gen);

        for line_size in LINE_SIZE_CANDIDATES {
            set = set.with(
                Tunable::new(
                    &format!("line_size_{line_size}"),
                    move |(weight_decay_base, weight_decay_input)| {
                        Ok::<_, String>(weight_decay_transform::<R, F>(
                            weight_decay_base,
                            weight_decay_input,
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
        &CubeTuneId::new(&weight_decay_input.client, &weight_decay_input.device),
        &client,
        tunables,
        (weight_decay_base, weight_decay_input),
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
    use crate::kernels::train::time_mixer::weight_decay_transform::WeightDecayTransformBackend;

    impl<B: FusionBackend + WeightDecayTransformBackend> WeightDecayTransformBackend for Fusion<B> {
        fn fused_weight_decay_transform(
            inputs: WeightDecayTransformForwardPrimitiveInputs<Self>,
        ) -> FloatTensor<Self> {
            let WeightDecayTransformForwardPrimitiveInputs {
                weight_decay_base,
                weight_decay_input,
            } = inputs;
            let client = weight_decay_input.client.clone();
            let [batch_size, context_len, embedded_dim] = weight_decay_input.shape.dims();

            #[derive(Clone, Debug)]
            struct WeightDecayTransformOp<B1> {
                desc: CustomOpIr,
                _backend: core::marker::PhantomData<B1>,
            }

            impl<B1: FusionBackend + WeightDecayTransformBackend> Operation<B1::FusionRuntime>
                for WeightDecayTransformOp<B1>
            {
                fn execute(
                    &self,
                    handles: &mut HandleContainer<
                        <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                    >,
                ) {
                    let ([weight_decay_base, weight_decay_input], [output_out]) =
                        self.desc.as_fixed();

                    let output = B1::fused_weight_decay_transform(
                        WeightDecayTransformForwardPrimitiveInputs {
                            weight_decay_base: handles.get_float_tensor::<B1>(weight_decay_base),
                            weight_decay_input: handles.get_float_tensor::<B1>(weight_decay_input),
                        },
                    );

                    handles.register_float_tensor::<B1>(&output_out.id, output);
                }
            }

            let mut streams = OperationStreams::default();
            streams.tensor(&weight_decay_base);
            streams.tensor(&weight_decay_input);

            let output_desc = [TensorIr::uninit(
                client.create_empty_handle(),
                Shape::new([batch_size, context_len, embedded_dim]),
                B::FloatElem::dtype(),
            )];

            let desc = CustomOpIr::new(
                "fused_weight_decay_transform",
                &[weight_decay_base.into_ir(), weight_decay_input.into_ir()],
                &output_desc,
            );

            let op = WeightDecayTransformOp::<B> {
                desc,
                _backend: core::marker::PhantomData,
            };

            client
                .register(streams, OperationIr::Custom(op.desc.clone()), op)
                .pop()
                .expect("missing fused_weight_decay_transform output")
        }
    }
}

fn weight_decay_transform<R, F>(
    weight_decay_base: CubeTensor<R>,
    weight_decay_input: CubeTensor<R>,
    vector_size: usize,
) -> CubeTensor<R>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
{
    let shape = weight_decay_input.meta.shape().clone();
    let client = weight_decay_input.client.clone();
    let output = empty_device::<R, F>(
        client.clone(),
        weight_decay_input.device.clone(),
        shape.clone(),
    );

    if shape.num_elements() == 0 {
        return output;
    }

    let working_units = shape.num_elements() / vector_size;
    let cube_dim = CubeDim::new(&client, working_units);
    let cube_count = calculate_cube_count_elemwise(&client, working_units, cube_dim);
    let address_type = max_address_type(&[&weight_decay_base, &weight_decay_input, &output]);

    unsafe {
        let embedded_vecs = shape[2] / vector_size;
        if embedded_vecs.is_power_of_two() {
            weight_decay_transform_forward_pow2_kernel::launch_unchecked::<F, R>(
                &client,
                cube_count,
                cube_dim,
                address_type,
                vector_size,
                weight_decay_base.into_linear_view(),
                weight_decay_input.into_linear_view_like(&output),
                output.clone().into_linear_view(),
                embedded_vecs - 1,
            );
        } else {
            weight_decay_transform_forward_kernel::launch_unchecked::<F, R>(
                &client,
                cube_count,
                cube_dim,
                address_type,
                vector_size,
                weight_decay_base.into_linear_view(),
                weight_decay_input.into_linear_view_like(&output),
                output.clone().into_linear_view(),
            );
        }
    }

    output
}

fn max_line_size_pair<R: CubeRuntime>(
    weight_decay_base: &CubeTensor<R>,
    weight_decay_input: &CubeTensor<R>,
) -> usize {
    let base_line_size = tensor_vector_size_parallel(
        weight_decay_base
            .client
            .io_optimized_vector_sizes(weight_decay_base.dtype.size()),
        weight_decay_base.meta.shape(),
        weight_decay_base.meta.strides(),
        0,
    );
    let input_line_size = tensor_vector_size_parallel(
        weight_decay_input
            .client
            .io_optimized_vector_sizes(weight_decay_input.dtype.size()),
        weight_decay_input.meta.shape(),
        weight_decay_input.meta.strides(),
        weight_decay_input.meta.shape().num_dims() - 1,
    );

    base_line_size.min(input_line_size).max(1)
}

fn max_address_type<R: CubeRuntime>(tensors: &[&CubeTensor<R>]) -> AddressType {
    tensors
        .iter()
        .map(|tensor| tensor.required_address_type())
        .max()
        .unwrap_or_default()
}
