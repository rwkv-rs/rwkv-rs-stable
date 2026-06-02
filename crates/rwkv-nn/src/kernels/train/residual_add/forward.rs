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

use crate::kernels::train::residual_add::{
    io::ResidualAddPrimitiveInputs,
    kernel::residual_add_forward_kernel,
};

const LINE_SIZE_CANDIDATES: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
struct ResidualAddForwardAutotuneKey {
    runtime: String,
    dtype: DType,
    num_elements: usize,
    rows: usize,
    innermost_dim: usize,
    load_width: u32,
    plane_size: u32,
    max_units_per_cube: u32,
    max_cube_dim: (u32, u32, u32),
    max_shared_memory_size: usize,
    max_vector_size: usize,
    num_streaming_multiprocessors: Option<u32>,
    num_tensor_cores: Option<u32>,
    min_tensor_cores_dim: Option<u32>,
    max_line_size: usize,
    lhs_in_place: bool,
    rhs_in_place: bool,
    deterministic: bool,
}

impl core::fmt::Display for ResidualAddForwardAutotuneKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{}:{:?}:n{}:r{}:d{}:load{}:p{}:u{}:cube{:?}:s{}:vec{}:sm{:?}:tc{:?}:tcdim{:?}:line{}:lhs{}:rhs{}:det{}",
            self.runtime,
            self.dtype,
            self.num_elements,
            self.rows,
            self.innermost_dim,
            self.load_width,
            self.plane_size,
            self.max_units_per_cube,
            self.max_cube_dim,
            self.max_shared_memory_size,
            self.max_vector_size,
            self.num_streaming_multiprocessors,
            self.num_tensor_cores,
            self.min_tensor_cores_dim,
            self.max_line_size,
            self.lhs_in_place,
            self.rhs_in_place,
            self.deterministic
        )
    }
}

impl AutotuneKey for ResidualAddForwardAutotuneKey {}

pub(crate) fn fused_residual_add<
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
>(
    inputs: ResidualAddPrimitiveInputs<CubeBackend<R, F, I, BT>>,
) -> FloatTensor<CubeBackend<R, F, I, BT>> {
    let ResidualAddPrimitiveInputs { lhs, rhs } = inputs;
    residual_add::<R, F, I, BT>(lhs, rhs)
}

fn residual_add<R, F, I, BT>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
) -> FloatTensor<CubeBackend<R, F, I, BT>>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
{
    let shape = lhs.meta.shape().clone();
    let client = lhs.client.clone();

    if shape.num_elements() == 0 {
        return lhs;
    }

    let key = |(lhs, rhs): &(CubeTensor<R>, CubeTensor<R>)| {
        let shape = lhs.meta.shape();
        let innermost_dim = shape[shape.num_dims() - 1];
        let hardware = &lhs.client.properties().hardware;

        ResidualAddForwardAutotuneKey {
            runtime: R::name(&lhs.client).to_owned(),
            dtype: lhs.dtype,
            num_elements: anchor(shape.num_elements(), None, Some(1), None),
            rows: anchor(shape.num_elements() / innermost_dim, None, Some(1), None),
            innermost_dim,
            load_width: hardware.load_width,
            plane_size: hardware.plane_size_max,
            max_units_per_cube: hardware.max_units_per_cube,
            max_cube_dim: hardware.max_cube_dim,
            max_shared_memory_size: hardware.max_shared_memory_size,
            max_vector_size: hardware.max_vector_size,
            num_streaming_multiprocessors: hardware.num_streaming_multiprocessors,
            num_tensor_cores: hardware.num_tensor_cores,
            min_tensor_cores_dim: hardware.min_tensor_cores_dim,
            max_line_size: max_line_size_many(&[lhs, rhs], shape.num_dims() - 1),
            lhs_in_place: lhs.can_mut() && lhs.is_nonoverlapping(),
            rhs_in_place: rhs.can_mut() && rhs.is_nonoverlapping(),
            deterministic: true,
        }
    };

    let input_gen = |_key: &ResidualAddForwardAutotuneKey,
                     (lhs, rhs): &(CubeTensor<R>, CubeTensor<R>)| {
        (lhs.copy(), rhs.copy())
    };

    static TUNER: LocalTuner<ResidualAddForwardAutotuneKey, CubeTuneId> =
        local_tuner!("residual-add-forward");

    let tunables = TUNER.init(move || {
        let line_size_group = TuneGroup::<ResidualAddForwardAutotuneKey>::new("line_size", |_| 1);
        let mut set = TunableSet::new(key, input_gen);

        for line_size in LINE_SIZE_CANDIDATES {
            set = set.with(
                Tunable::new(&format!("line_size_{line_size}"), move |(lhs, rhs)| {
                    Ok::<_, String>(residual_add_with_vector::<R, F, I, BT>(lhs, rhs, line_size))
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
        &CubeTuneId::new(&lhs.client, &lhs.device),
        &client,
        tunables,
        (lhs, rhs),
    )
}

fn residual_add_with_vector<R, F, I, BT>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    vector_size: usize,
) -> FloatTensor<CubeBackend<R, F, I, BT>>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
{
    let shape = lhs.meta.shape().clone();
    let client = lhs.client.clone();
    let working_units = shape.num_elements() / vector_size;
    let cube_dim = CubeDim::new(&client, working_units);
    let cube_count = calculate_cube_count_elemwise(&client, working_units, cube_dim);
    let lhs_can_mut = lhs.can_mut() && lhs.is_nonoverlapping();
    let rhs_can_mut = rhs.can_mut() && rhs.is_nonoverlapping();

    unsafe {
        if lhs_can_mut {
            residual_add_forward_kernel::launch_unchecked::<F, R>(
                &client,
                cube_count,
                cube_dim,
                max_address_type(&[&lhs, &rhs]),
                vector_size,
                lhs.clone().into_linear_view(),
                rhs.into_linear_view_like(&lhs),
                lhs.as_linear_view_alias(0),
            );

            lhs
        } else if rhs_can_mut {
            residual_add_forward_kernel::launch_unchecked::<F, R>(
                &client,
                cube_count,
                cube_dim,
                max_address_type(&[&lhs, &rhs]),
                vector_size,
                lhs.into_linear_view_like(&rhs),
                rhs.clone().into_linear_view(),
                rhs.as_linear_view_alias(1),
            );

            rhs
        } else {
            let output = empty_device::<R, F>(client.clone(), lhs.device.clone(), shape);

            residual_add_forward_kernel::launch_unchecked::<F, R>(
                &client,
                cube_count,
                cube_dim,
                max_address_type(&[&lhs, &rhs, &output]),
                vector_size,
                lhs.into_linear_view_like(&output),
                rhs.into_linear_view_like(&output),
                output.clone().into_linear_view(),
            );

            output
        }
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
    use crate::kernels::train::residual_add::ResidualAddBackend;

    impl<B: FusionBackend + ResidualAddBackend> ResidualAddBackend for Fusion<B> {
        fn fused_residual_add(inputs: ResidualAddPrimitiveInputs<Self>) -> FloatTensor<Self> {
            let ResidualAddPrimitiveInputs { lhs, rhs } = inputs;
            let client = lhs.client.clone();
            let shape = lhs.shape.clone();

            #[derive(Clone, Debug)]
            struct ResidualAddOp<B1> {
                desc: CustomOpIr,
                _backend: core::marker::PhantomData<B1>,
            }

            impl<B1: FusionBackend + ResidualAddBackend> Operation<B1::FusionRuntime> for ResidualAddOp<B1> {
                fn execute(
                    &self,
                    handles: &mut HandleContainer<
                        <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                    >,
                ) {
                    let ([lhs, rhs], [output_out]) = self.desc.as_fixed();

                    let output = B1::fused_residual_add(ResidualAddPrimitiveInputs {
                        lhs: handles.get_float_tensor::<B1>(lhs),
                        rhs: handles.get_float_tensor::<B1>(rhs),
                    });

                    handles.register_float_tensor::<B1>(&output_out.id, output);
                }
            }

            let mut streams = OperationStreams::default();
            streams.tensor(&lhs);
            streams.tensor(&rhs);

            let output_desc = [TensorIr::uninit(
                client.create_empty_handle(),
                shape,
                B::FloatElem::dtype(),
            )];

            let desc = CustomOpIr::new(
                "fused_residual_add",
                &[lhs.into_ir(), rhs.into_ir()],
                &output_desc,
            );

            let op = ResidualAddOp::<B> {
                desc,
                _backend: core::marker::PhantomData,
            };

            client
                .register(streams, OperationIr::Custom(op.desc.clone()), op)
                .pop()
                .expect("missing fused_residual_add output")
        }
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
