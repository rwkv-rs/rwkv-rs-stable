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

use crate::kernels::train::time_mixer::learning_rate_gate::{
    io::LearningRateGateForwardPrimitiveInputs,
    kernel::{learning_rate_gate_forward_kernel, learning_rate_gate_forward_pow2_kernel},
};

const LINE_SIZE_CANDIDATES: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
struct LearningRateGateForwardAutotuneKey {
    dtype: DType,
    num_elements: usize,
    embedded_dim: usize,
    max_line_size: usize,
}

impl core::fmt::Display for LearningRateGateForwardAutotuneKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{:?}:{}:{}:{}",
            self.dtype, self.num_elements, self.embedded_dim, self.max_line_size
        )
    }
}

impl AutotuneKey for LearningRateGateForwardAutotuneKey {}

pub(crate) fn fused_learning_rate_gate<
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
>(
    inputs: LearningRateGateForwardPrimitiveInputs<CubeBackend<R, F, I, BT>>,
) -> FloatTensor<CubeBackend<R, F, I, BT>> {
    let LearningRateGateForwardPrimitiveInputs {
        learning_rate_base,
        learning_rate_input,
    } = inputs;
    let client = learning_rate_input.client.clone();

    let key = |learning_rate_base: &CubeTensor<R>, learning_rate_input: &CubeTensor<R>| {
        let shape = learning_rate_input.meta.shape();

        LearningRateGateForwardAutotuneKey {
            dtype: learning_rate_input.dtype,
            num_elements: anchor(shape.num_elements(), None, Some(1), None),
            embedded_dim: shape[2],
            max_line_size: max_line_size_pair(learning_rate_base, learning_rate_input),
        }
    };

    let input_gen = |_key: &LearningRateGateForwardAutotuneKey,
                     learning_rate_base: &CubeTensor<R>,
                     learning_rate_input: &CubeTensor<R>| {
        (learning_rate_base.copy(), learning_rate_input.copy())
    };

    static TUNER: LocalTuner<LearningRateGateForwardAutotuneKey, CubeTuneId> =
        local_tuner!("learning-rate-gate-forward");

    let tunables = TUNER.init(move || {
        let line_size_group =
            TuneGroup::<LearningRateGateForwardAutotuneKey>::new("line_size", |_| 1);
        let mut set = TunableSet::new(key, input_gen);

        for line_size in LINE_SIZE_CANDIDATES {
            set = set.with(
                Tunable::new(
                    format!("line_size_{line_size}"),
                    (move |learning_rate_base: CubeTensor<R>,
                           learning_rate_input: CubeTensor<R>| {
                        learning_rate_gate::<R, F>(
                            learning_rate_base,
                            learning_rate_input,
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
        &CubeTuneId::new(&learning_rate_input.client, &learning_rate_input.device),
        &client,
        tunables,
        (learning_rate_base, learning_rate_input),
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
    use crate::kernels::train::time_mixer::learning_rate_gate::LearningRateGateBackend;

    impl<B: FusionBackend + LearningRateGateBackend> LearningRateGateBackend for Fusion<B> {
        fn fused_learning_rate_gate(
            inputs: LearningRateGateForwardPrimitiveInputs<Self>,
        ) -> FloatTensor<Self> {
            let LearningRateGateForwardPrimitiveInputs {
                learning_rate_base,
                learning_rate_input,
            } = inputs;
            let client = learning_rate_input.client.clone();
            let [batch_size, context_len, embedded_dim] = learning_rate_input.shape.dims();

            #[derive(Clone, Debug)]
            struct LearningRateGateOp<B1> {
                desc: CustomOpIr,
                _backend: core::marker::PhantomData<B1>,
            }

            impl<B1: FusionBackend + LearningRateGateBackend> Operation<B1::FusionRuntime>
                for LearningRateGateOp<B1>
            {
                fn execute(
                    &self,
                    handles: &mut HandleContainer<
                        <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                    >,
                ) {
                    let ([learning_rate_base, learning_rate_input], [learning_rate_output_out]) =
                        self.desc.as_fixed();

                    let output =
                        B1::fused_learning_rate_gate(LearningRateGateForwardPrimitiveInputs {
                            learning_rate_base: handles.get_float_tensor::<B1>(learning_rate_base),
                            learning_rate_input: handles
                                .get_float_tensor::<B1>(learning_rate_input),
                        });

                    handles.register_float_tensor::<B1>(&learning_rate_output_out.id, output);
                }
            }

            let mut streams = OperationStreams::default();
            streams.tensor(&learning_rate_base);
            streams.tensor(&learning_rate_input);

            let output_desc = [TensorIr::uninit(
                client.create_empty_handle(),
                Shape::new([batch_size, context_len, embedded_dim]),
                B::FloatElem::dtype(),
            )];

            let desc = CustomOpIr::new(
                "fused_learning_rate_gate",
                &[learning_rate_base.into_ir(), learning_rate_input.into_ir()],
                &output_desc,
            );

            let op = LearningRateGateOp::<B> {
                desc,
                _backend: core::marker::PhantomData,
            };

            client
                .register(streams, OperationIr::Custom(op.desc.clone()), op)
                .pop()
                .expect("missing fused_learning_rate_gate output")
        }
    }
}

fn learning_rate_gate<R, F>(
    learning_rate_base: CubeTensor<R>,
    learning_rate_input: CubeTensor<R>,
    vector_size: usize,
) -> CubeTensor<R>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
{
    let shape = learning_rate_input.meta.shape().clone();
    let client = learning_rate_input.client.clone();
    let output = empty_device::<R, F>(
        client.clone(),
        learning_rate_input.device.clone(),
        shape.clone(),
    );

    if shape.num_elements() == 0 {
        return output;
    }

    let working_units = shape.num_elements() / vector_size;
    let cube_dim = CubeDim::new(&client, working_units);
    let cube_count = calculate_cube_count_elemwise(&client, working_units, cube_dim);
    let address_type = max_address_type(&[&learning_rate_base, &learning_rate_input, &output]);

    // The forward equation is elementwise after broadcasting `learning_rate_base` across
    // `[batch_size, context_len]`. Each work unit computes one vector of contiguous
    // embedded-dimension lanes, so memory reads and writes are coalesced when the tuned vector
    // width divides `embedded_dim`. Live state is the base vector, input vector, sigmoid
    // temporaries, and output vector. The small base tensor is repeatedly indexed by embedded
    // dimension and relies on normal cache reuse.
    // SAFETY: The public input contract checks dtype/device/shape/contiguity. Autotune only keeps
    // vector sizes that divide the embedded dimension, so the linear vector views cover the logical
    // element range exactly.
    unsafe {
        let embedded_vecs = shape[2] / vector_size;
        if embedded_vecs.is_power_of_two() {
            learning_rate_gate_forward_pow2_kernel::launch_unchecked::<F, R>(
                &client,
                cube_count,
                cube_dim,
                address_type,
                vector_size,
                learning_rate_base.into_linear_view(),
                learning_rate_input.into_linear_view_like(&output),
                output.clone().into_linear_view(),
                embedded_vecs - 1,
            );
        } else {
            learning_rate_gate_forward_kernel::launch_unchecked::<F, R>(
                &client,
                cube_count,
                cube_dim,
                address_type,
                vector_size,
                learning_rate_base.into_linear_view(),
                learning_rate_input.into_linear_view_like(&output),
                output.clone().into_linear_view(),
            );
        }
    }

    output
}

fn max_line_size_pair<R: CubeRuntime>(
    learning_rate_base: &CubeTensor<R>,
    learning_rate_input: &CubeTensor<R>,
) -> usize {
    let base_line_size = tensor_vector_size_parallel(
        learning_rate_base
            .client
            .io_optimized_vector_sizes(learning_rate_base.dtype.size()),
        learning_rate_base.meta.shape(),
        learning_rate_base.meta.strides(),
        0,
    );
    let input_line_size = tensor_vector_size_parallel(
        learning_rate_input
            .client
            .io_optimized_vector_sizes(learning_rate_input.dtype.size()),
        learning_rate_input.meta.shape(),
        learning_rate_input.meta.strides(),
        learning_rate_input.meta.shape().num_dims() - 1,
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
