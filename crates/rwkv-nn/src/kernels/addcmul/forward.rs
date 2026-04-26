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

use crate::kernels::addcmul::{
    io::{
        Addcmul5ForwardPrimitiveInputs,
        Addcmul5ForwardPrimitiveOutput,
        AddcmulForwardPrimitiveInputs,
    },
    kernel::{
        Addcmul5ForwardInputsLaunch,
        Addcmul5ForwardOutputsLaunch,
        addcmul_forward_kernel,
        addcmul5_forward_kernel,
    },
};

const LINE_SIZE_CANDIDATES: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
enum AddcmulForwardOp {
    Addcmul,
    Addcmul5,
}

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
struct AddcmulForwardAutotuneKey {
    op: AddcmulForwardOp,
    dtype: DType,
    num_elements: usize,
    embedded_dim: usize,
    max_line_size: usize,
    can_mut: bool,
}

impl core::fmt::Display for AddcmulForwardAutotuneKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{:?}:{:?}:{}:{}:{}:{}",
            self.op,
            self.dtype,
            self.num_elements,
            self.embedded_dim,
            self.max_line_size,
            self.can_mut
        )
    }
}

impl AutotuneKey for AddcmulForwardAutotuneKey {}

impl<R, F, I, BT> AutotuneOutput for Addcmul5ForwardPrimitiveOutput<CubeBackend<R, F, I, BT>>
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
    }
}

pub(crate) fn fused_addcmul<
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
>(
    inputs: AddcmulForwardPrimitiveInputs<CubeBackend<R, F, I, BT>>,
) -> FloatTensor<CubeBackend<R, F, I, BT>> {
    let AddcmulForwardPrimitiveInputs { base, diff, scale } = inputs;
    let client = base.client.clone();

    let key = |base: &CubeTensor<R>, diff: &CubeTensor<R>, scale: &CubeTensor<R>| {
        let shape = base.meta.shape();

        AddcmulForwardAutotuneKey {
            op: AddcmulForwardOp::Addcmul,
            dtype: base.dtype,
            num_elements: anchor(shape.num_elements(), None, Some(1), None),
            embedded_dim: shape[2],
            max_line_size: max_line_size_many(&[base, diff, scale], shape.num_dims() - 1),
            can_mut: base.can_mut() && base.is_nonoverlapping(),
        }
    };

    let input_gen =
        |_key: &AddcmulForwardAutotuneKey,
         base: &CubeTensor<R>,
         diff: &CubeTensor<R>,
         scale: &CubeTensor<R>| { (base.copy(), diff.copy(), scale.copy()) };

    static TUNER: LocalTuner<AddcmulForwardAutotuneKey, CubeTuneId> =
        local_tuner!("addcmul-forward");

    let tunables = TUNER.init(move || {
        let line_size_group = TuneGroup::<AddcmulForwardAutotuneKey>::new("line_size", |_| 1);
        let mut set = TunableSet::new(key, input_gen);

        for line_size in LINE_SIZE_CANDIDATES {
            set = set.with(
                Tunable::new(
                    format!("line_size_{line_size}"),
                    (move |base: CubeTensor<R>, diff: CubeTensor<R>, scale: CubeTensor<R>| {
                        addcmul::<R, F>(base, diff, scale, line_size)
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
        &CubeTuneId::new(&base.client, &base.device),
        &client,
        tunables,
        (base, diff, scale),
    )
}

pub(crate) fn fused_addcmul5<
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
>(
    inputs: Addcmul5ForwardPrimitiveInputs<CubeBackend<R, F, I, BT>>,
) -> Addcmul5ForwardPrimitiveOutput<CubeBackend<R, F, I, BT>> {
    let Addcmul5ForwardPrimitiveInputs {
        base,
        diff,
        receptance_scale,
        weight_decay_scale,
        key_scale,
        value_scale,
        learning_rate_scale,
    } = inputs;
    let client = base.client.clone();

    let key = |base: &CubeTensor<R>,
               diff: &CubeTensor<R>,
               receptance_scale: &CubeTensor<R>,
               weight_decay_scale: &CubeTensor<R>,
               key_scale: &CubeTensor<R>,
               value_scale: &CubeTensor<R>,
               learning_rate_scale: &CubeTensor<R>| {
        let shape = base.meta.shape();

        AddcmulForwardAutotuneKey {
            op: AddcmulForwardOp::Addcmul5,
            dtype: base.dtype,
            num_elements: anchor(shape.num_elements(), None, Some(1), None),
            embedded_dim: shape[2],
            max_line_size: max_line_size_many(
                &[
                    base,
                    diff,
                    receptance_scale,
                    weight_decay_scale,
                    key_scale,
                    value_scale,
                    learning_rate_scale,
                ],
                shape.num_dims() - 1,
            ),
            can_mut: false,
        }
    };

    let input_gen = |_key: &AddcmulForwardAutotuneKey,
                     base: &CubeTensor<R>,
                     diff: &CubeTensor<R>,
                     receptance_scale: &CubeTensor<R>,
                     weight_decay_scale: &CubeTensor<R>,
                     key_scale: &CubeTensor<R>,
                     value_scale: &CubeTensor<R>,
                     learning_rate_scale: &CubeTensor<R>| {
        (
            base.copy(),
            diff.copy(),
            receptance_scale.copy(),
            weight_decay_scale.copy(),
            key_scale.copy(),
            value_scale.copy(),
            learning_rate_scale.copy(),
        )
    };

    static TUNER: LocalTuner<AddcmulForwardAutotuneKey, CubeTuneId> =
        local_tuner!("addcmul5-forward");

    let tunables = TUNER.init(move || {
        let line_size_group = TuneGroup::<AddcmulForwardAutotuneKey>::new("line_size", |_| 1);
        let mut set = TunableSet::new(key, input_gen);

        for line_size in LINE_SIZE_CANDIDATES {
            set = set.with(
                Tunable::new(
                    format!("line_size_{line_size}"),
                    (move |base: CubeTensor<R>,
                           diff: CubeTensor<R>,
                           receptance_scale: CubeTensor<R>,
                           weight_decay_scale: CubeTensor<R>,
                           key_scale: CubeTensor<R>,
                           value_scale: CubeTensor<R>,
                           learning_rate_scale: CubeTensor<R>| {
                        addcmul5::<R, F, I, BT>(
                            base,
                            diff,
                            receptance_scale,
                            weight_decay_scale,
                            key_scale,
                            value_scale,
                            learning_rate_scale,
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
        &CubeTuneId::new(&base.client, &base.device),
        &client,
        tunables,
        (
            base,
            diff,
            receptance_scale,
            weight_decay_scale,
            key_scale,
            value_scale,
            learning_rate_scale,
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
    use crate::kernels::addcmul::AddcmulBackend;

    impl<B: FusionBackend + AddcmulBackend> AddcmulBackend for Fusion<B> {
        fn fused_addcmul(inputs: AddcmulForwardPrimitiveInputs<Self>) -> FloatTensor<Self> {
            let AddcmulForwardPrimitiveInputs { base, diff, scale } = inputs;
            let client = base.client.clone();
            let [batch_size, context_length, embedded_dim] = base.shape.dims();

            #[derive(Clone, Debug)]
            struct AddcmulOp<B1> {
                desc: CustomOpIr,
                _backend: core::marker::PhantomData<B1>,
            }

            impl<B1: FusionBackend + AddcmulBackend> Operation<B1::FusionRuntime> for AddcmulOp<B1> {
                fn execute(
                    &self,
                    handles: &mut HandleContainer<
                        <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                    >,
                ) {
                    let ([base, diff, scale], [output_out]) = self.desc.as_fixed();

                    let output = B1::fused_addcmul(AddcmulForwardPrimitiveInputs {
                        base: handles.get_float_tensor::<B1>(base),
                        diff: handles.get_float_tensor::<B1>(diff),
                        scale: handles.get_float_tensor::<B1>(scale),
                    });

                    handles.register_float_tensor::<B1>(&output_out.id, output);
                }
            }

            let mut streams = OperationStreams::default();
            streams.tensor(&base);
            streams.tensor(&diff);
            streams.tensor(&scale);

            let output_desc = [TensorIr::uninit(
                client.create_empty_handle(),
                Shape::new([batch_size, context_length, embedded_dim]),
                B::FloatElem::dtype(),
            )];

            let desc = CustomOpIr::new(
                "fused_addcmul",
                &[base.into_ir(), diff.into_ir(), scale.into_ir()],
                &output_desc,
            );

            let op = AddcmulOp::<B> {
                desc,
                _backend: core::marker::PhantomData,
            };

            client
                .register(streams, OperationIr::Custom(op.desc.clone()), op)
                .pop()
                .expect("missing fused_addcmul output")
        }

        fn fused_addcmul5(
            inputs: Addcmul5ForwardPrimitiveInputs<Self>,
        ) -> Addcmul5ForwardPrimitiveOutput<Self> {
            let Addcmul5ForwardPrimitiveInputs {
                base,
                diff,
                receptance_scale,
                weight_decay_scale,
                key_scale,
                value_scale,
                learning_rate_scale,
            } = inputs;
            let client = base.client.clone();
            let [batch_size, context_length, embedded_dim] = base.shape.dims();

            #[derive(Clone, Debug)]
            struct Addcmul5Op<B1> {
                desc: CustomOpIr,
                _backend: core::marker::PhantomData<B1>,
            }

            impl<B1: FusionBackend + AddcmulBackend> Operation<B1::FusionRuntime> for Addcmul5Op<B1> {
                fn execute(
                    &self,
                    handles: &mut HandleContainer<
                        <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                    >,
                ) {
                    let (
                        [
                            base,
                            diff,
                            receptance_scale,
                            weight_decay_scale,
                            key_scale,
                            value_scale,
                            learning_rate_scale,
                        ],
                        [
                            receptance_output_out,
                            weight_decay_output_out,
                            key_output_out,
                            value_output_out,
                            learning_rate_output_out,
                        ],
                    ) = self.desc.as_fixed();

                    let output = B1::fused_addcmul5(Addcmul5ForwardPrimitiveInputs {
                        base: handles.get_float_tensor::<B1>(base),
                        diff: handles.get_float_tensor::<B1>(diff),
                        receptance_scale: handles.get_float_tensor::<B1>(receptance_scale),
                        weight_decay_scale: handles.get_float_tensor::<B1>(weight_decay_scale),
                        key_scale: handles.get_float_tensor::<B1>(key_scale),
                        value_scale: handles.get_float_tensor::<B1>(value_scale),
                        learning_rate_scale: handles.get_float_tensor::<B1>(learning_rate_scale),
                    });

                    handles.register_float_tensor::<B1>(
                        &receptance_output_out.id,
                        output.receptance_input,
                    );
                    handles.register_float_tensor::<B1>(
                        &weight_decay_output_out.id,
                        output.weight_decay_input,
                    );
                    handles.register_float_tensor::<B1>(&key_output_out.id, output.key_input);
                    handles.register_float_tensor::<B1>(&value_output_out.id, output.value_input);
                    handles.register_float_tensor::<B1>(
                        &learning_rate_output_out.id,
                        output.learning_rate_input,
                    );
                }
            }

            let mut streams = OperationStreams::default();
            streams.tensor(&base);
            streams.tensor(&diff);
            streams.tensor(&receptance_scale);
            streams.tensor(&weight_decay_scale);
            streams.tensor(&key_scale);
            streams.tensor(&value_scale);
            streams.tensor(&learning_rate_scale);

            let output_desc = [
                TensorIr::uninit(
                    client.create_empty_handle(),
                    Shape::new([batch_size, context_length, embedded_dim]),
                    B::FloatElem::dtype(),
                ),
                TensorIr::uninit(
                    client.create_empty_handle(),
                    Shape::new([batch_size, context_length, embedded_dim]),
                    B::FloatElem::dtype(),
                ),
                TensorIr::uninit(
                    client.create_empty_handle(),
                    Shape::new([batch_size, context_length, embedded_dim]),
                    B::FloatElem::dtype(),
                ),
                TensorIr::uninit(
                    client.create_empty_handle(),
                    Shape::new([batch_size, context_length, embedded_dim]),
                    B::FloatElem::dtype(),
                ),
                TensorIr::uninit(
                    client.create_empty_handle(),
                    Shape::new([batch_size, context_length, embedded_dim]),
                    B::FloatElem::dtype(),
                ),
            ];

            let desc = CustomOpIr::new(
                "fused_addcmul5",
                &[
                    base.into_ir(),
                    diff.into_ir(),
                    receptance_scale.into_ir(),
                    weight_decay_scale.into_ir(),
                    key_scale.into_ir(),
                    value_scale.into_ir(),
                    learning_rate_scale.into_ir(),
                ],
                &output_desc,
            );

            let op = Addcmul5Op::<B> {
                desc,
                _backend: core::marker::PhantomData,
            };

            let mut outputs = client.register(streams, OperationIr::Custom(op.desc.clone()), op);

            let learning_rate_input = outputs.pop().expect("missing learning_rate_input");
            let value_input = outputs.pop().expect("missing value_input");
            let key_input = outputs.pop().expect("missing key_input");
            let weight_decay_input = outputs.pop().expect("missing weight_decay_input");
            let receptance_input = outputs.pop().expect("missing receptance_input");

            Addcmul5ForwardPrimitiveOutput {
                receptance_input,
                weight_decay_input,
                key_input,
                value_input,
                learning_rate_input,
            }
        }
    }
}

fn addcmul<R, F>(
    base: CubeTensor<R>,
    diff: CubeTensor<R>,
    scale: CubeTensor<R>,
    vector_size: usize,
) -> CubeTensor<R>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
{
    let shape = base.meta.shape().clone();
    let client = base.client.clone();
    let output = if base.can_mut() && base.is_nonoverlapping() {
        base.clone()
    } else {
        empty_device::<R, F>(client.clone(), base.device.clone(), shape.clone())
    };

    if shape.num_elements() == 0 {
        return output;
    }

    let working_units = shape.num_elements() / vector_size;
    let cube_dim = CubeDim::new(&client, working_units);
    let cube_count = calculate_cube_count_elemwise(&client, working_units, cube_dim);
    let address_type = max_address_type(&[&base, &diff, &scale]);

    // The forward equation is fully elementwise, so the launch exposes one independent work unit
    // per `vector_size` contiguous elements. Autotune chooses `vector_size` from line-size
    // candidates supported by the device and divisible by `embedded_dim`; this keeps vector lanes
    // aligned with the last tensor axis and preserves coalesced reads/writes for `base`, `diff`,
    // and `output`. Each work unit only needs the current base vector, diff vector, scale vector,
    // and output vector live, so register pressure stays bounded by the chosen vector width. The
    // scale tensor is only `[1, 1, embedded_dim]`; adjacent context positions reuse the same scale
    // addresses, leaving L1/L2 reuse to the hardware cache instead of staging shared memory.
    // SAFETY: The public addcmul input contract checks dtype/device/shape/contiguity before
    // primitive dispatch. Autotune only enables vector sizes that divide the embedding axis,
    // so these linear vector views cover the same logical element range.
    unsafe {
        if base.can_mut() && base.is_nonoverlapping() {
            addcmul_forward_kernel::launch_unchecked::<F, R>(
                &client,
                cube_count,
                cube_dim,
                address_type,
                vector_size,
                base.clone().into_linear_view(),
                diff.into_linear_view_like(&base),
                scale.into_linear_view(),
                base.as_linear_view_alias(0),
            );
        } else {
            addcmul_forward_kernel::launch_unchecked::<F, R>(
                &client,
                cube_count,
                cube_dim,
                address_type.max(output.required_address_type()),
                vector_size,
                base.into_linear_view_like(&output),
                diff.into_linear_view_like(&output),
                scale.into_linear_view(),
                output.clone().into_linear_view(),
            );
        }
    }

    output
}

fn addcmul5<R, F, I, BT>(
    base: CubeTensor<R>,
    diff: CubeTensor<R>,
    receptance_scale: CubeTensor<R>,
    weight_decay_scale: CubeTensor<R>,
    key_scale: CubeTensor<R>,
    value_scale: CubeTensor<R>,
    learning_rate_scale: CubeTensor<R>,
    vector_size: usize,
) -> Addcmul5ForwardPrimitiveOutput<CubeBackend<R, F, I, BT>>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
{
    let shape = base.meta.shape().clone();
    let client = base.client.clone();
    let receptance_output =
        empty_device::<R, F>(client.clone(), base.device.clone(), shape.clone());
    let weight_decay_output =
        empty_device::<R, F>(client.clone(), base.device.clone(), shape.clone());
    let key_output = empty_device::<R, F>(client.clone(), base.device.clone(), shape.clone());
    let value_output = empty_device::<R, F>(client.clone(), base.device.clone(), shape.clone());
    let learning_rate_output =
        empty_device::<R, F>(client.clone(), base.device.clone(), shape.clone());

    if shape.num_elements() == 0 {
        return Addcmul5ForwardPrimitiveOutput {
            receptance_input: receptance_output,
            weight_decay_input: weight_decay_output,
            key_input: key_output,
            value_input: value_output,
            learning_rate_input: learning_rate_output,
        };
    }

    let working_units = shape.num_elements() / vector_size;
    let cube_dim = CubeDim::new(&client, working_units);
    let cube_count = calculate_cube_count_elemwise(&client, working_units, cube_dim);
    let address_type = max_address_type(&[
        &base,
        &diff,
        &receptance_scale,
        &weight_decay_scale,
        &key_scale,
        &value_scale,
        &learning_rate_scale,
    ])
    .max(receptance_output.required_address_type())
    .max(weight_decay_output.required_address_type())
    .max(key_output.required_address_type())
    .max(value_output.required_address_type())
    .max(learning_rate_output.required_address_type());

    // The five-output kernel uses the same elementwise parallelization as `addcmul`, but each
    // work unit computes five branch formulas for the same vector position. That trades higher
    // per-thread live state for less global memory traffic: `base` and `diff` are loaded once and
    // reused with five scale vectors before writing five output vectors. This is why vectorized
    // linear views are useful here; a scalar implementation would expose the same independence but
    // would leave bandwidth on the table for contiguous `embedded_dim` lanes. The scale tensors are
    // small and repeatedly indexed by embedded dimension, so cache reuse is expected from normal
    // L1/L2 behavior rather than from explicit tiling.
    // SAFETY: The public addcmul5 input contract checks dtype/device/shape/contiguity before
    // primitive dispatch. Autotune only enables vector sizes that divide the embedding axis, and
    // each output is allocated with the same shape as `base`.
    unsafe {
        addcmul5_forward_kernel::launch_unchecked::<F, R>(
            &client,
            cube_count,
            cube_dim,
            address_type,
            vector_size,
            Addcmul5ForwardInputsLaunch::new(
                base.into_linear_view_like(&receptance_output),
                diff.into_linear_view_like(&receptance_output),
                receptance_scale.into_linear_view(),
                weight_decay_scale.into_linear_view(),
                key_scale.into_linear_view(),
                value_scale.into_linear_view(),
                learning_rate_scale.into_linear_view(),
            ),
            Addcmul5ForwardOutputsLaunch::new(
                receptance_output.clone().into_linear_view(),
                weight_decay_output.clone().into_linear_view(),
                key_output.clone().into_linear_view(),
                value_output.clone().into_linear_view(),
                learning_rate_output.clone().into_linear_view(),
            ),
        );
    }

    Addcmul5ForwardPrimitiveOutput {
        receptance_input: receptance_output,
        weight_decay_input: weight_decay_output,
        key_input: key_output,
        value_input: value_output,
        learning_rate_input: learning_rate_output,
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
