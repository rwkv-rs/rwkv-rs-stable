use burn::tensor::{Shape, ops::FloatTensor};
use burn_cubecl::{
    CubeBackend,
    CubeElement,
    CubeRuntime,
    FloatElement,
    IntElement,
    cubecl::{CubeCount, CubeDim, prelude::*},
    element::BoolElement,
    ops::numeric::empty_device,
    tensor::CubeTensor,
};

use crate::kernels::train::time_mixer::gated_readout_combine::{
    io::GatedReadoutCombineForwardPrimitiveInputs,
    kernel::gated_readout_combine_forward_kernel,
};

const HEAD_SIZE: usize = 64;
const BLOCK_SIZE: u32 = 64;
const NUM_WARPS: usize = 2;

pub(crate) fn fused_gated_readout_combine<
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
>(
    inputs: GatedReadoutCombineForwardPrimitiveInputs<CubeBackend<R, F, I, BT>>,
) -> FloatTensor<CubeBackend<R, F, I, BT>> {
    let GatedReadoutCombineForwardPrimitiveInputs {
        wkv_output,
        norm_gamma,
        norm_beta,
        norm_epsilon,
        gate,
        receptance,
        replacement_key,
        value,
        bonus,
    } = inputs;

    gated_readout_combine::<R, F>(
        wkv_output,
        norm_gamma,
        norm_beta,
        norm_epsilon as f32,
        gate,
        receptance,
        replacement_key,
        value,
        bonus,
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
    use crate::kernels::train::time_mixer::gated_readout_combine::GatedReadoutCombineBackend;

    impl<B: FusionBackend + GatedReadoutCombineBackend> GatedReadoutCombineBackend for Fusion<B> {
        fn fused_gated_readout_combine(
            inputs: GatedReadoutCombineForwardPrimitiveInputs<Self>,
        ) -> FloatTensor<Self> {
            let GatedReadoutCombineForwardPrimitiveInputs {
                wkv_output,
                norm_gamma,
                norm_beta,
                norm_epsilon,
                gate,
                receptance,
                replacement_key,
                value,
                bonus,
            } = inputs;
            let client = wkv_output.client.clone();
            let [batch_size, context_len, num_heads, head_size] = wkv_output.shape.dims();
            let embedded_dim = num_heads * head_size;

            #[derive(Clone, Debug)]
            struct GatedReadoutCombineOp<B1> {
                desc: CustomOpIr,
                norm_epsilon: f64,
                _backend: core::marker::PhantomData<B1>,
            }

            impl<B1: FusionBackend + GatedReadoutCombineBackend> Operation<B1::FusionRuntime>
                for GatedReadoutCombineOp<B1>
            {
                fn execute(
                    &self,
                    handles: &mut HandleContainer<
                        <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                    >,
                ) {
                    let (
                        [
                            wkv_output,
                            norm_gamma,
                            norm_beta,
                            gate,
                            receptance,
                            replacement_key,
                            value,
                            bonus,
                        ],
                        [output_out],
                    ) = self.desc.as_fixed();

                    let output = B1::fused_gated_readout_combine(
                        GatedReadoutCombineForwardPrimitiveInputs {
                            wkv_output: handles.get_float_tensor::<B1>(wkv_output),
                            norm_gamma: handles.get_float_tensor::<B1>(norm_gamma),
                            norm_beta: handles.get_float_tensor::<B1>(norm_beta),
                            norm_epsilon: self.norm_epsilon,
                            gate: handles.get_float_tensor::<B1>(gate),
                            receptance: handles.get_float_tensor::<B1>(receptance),
                            replacement_key: handles.get_float_tensor::<B1>(replacement_key),
                            value: handles.get_float_tensor::<B1>(value),
                            bonus: handles.get_float_tensor::<B1>(bonus),
                        },
                    );

                    handles.register_float_tensor::<B1>(&output_out.id, output);
                }
            }

            let mut streams = OperationStreams::default();
            streams.tensor(&wkv_output);
            streams.tensor(&norm_gamma);
            streams.tensor(&norm_beta);
            streams.tensor(&gate);
            streams.tensor(&receptance);
            streams.tensor(&replacement_key);
            streams.tensor(&value);
            streams.tensor(&bonus);

            let output_desc = [TensorIr::uninit(
                client.create_empty_handle(),
                Shape::new([batch_size, context_len, embedded_dim]),
                B::FloatElem::dtype(),
            )];

            let desc = CustomOpIr::new(
                "fused_gated_readout_combine",
                &[
                    wkv_output.into_ir(),
                    norm_gamma.into_ir(),
                    norm_beta.into_ir(),
                    gate.into_ir(),
                    receptance.into_ir(),
                    replacement_key.into_ir(),
                    value.into_ir(),
                    bonus.into_ir(),
                ],
                &output_desc,
            );

            let op = GatedReadoutCombineOp::<B> {
                desc,
                norm_epsilon,
                _backend: core::marker::PhantomData,
            };

            client
                .register(streams, OperationIr::Custom(op.desc.clone()), op)
                .pop()
                .expect("missing fused_gated_readout_combine output")
        }
    }
}

fn gated_readout_combine<R, F>(
    wkv_output: CubeTensor<R>,
    norm_gamma: CubeTensor<R>,
    norm_beta: CubeTensor<R>,
    norm_epsilon: f32,
    gate: CubeTensor<R>,
    receptance: CubeTensor<R>,
    replacement_key: CubeTensor<R>,
    value: CubeTensor<R>,
    bonus: CubeTensor<R>,
) -> CubeTensor<R>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
{
    let shape = wkv_output.meta.shape().clone();
    let client = wkv_output.client.clone();
    let [rows, num_heads, head_size, embedded_dim] = flatten_4d_shape(&shape);
    let output_shape = Shape::new([shape[0], shape[1], embedded_dim]);
    let output = empty_device::<R, F>(client.clone(), wkv_output.device.clone(), output_shape);

    if shape.num_elements() == 0 {
        return output;
    }

    let address_type = max_address_type(&[
        &wkv_output,
        &norm_gamma,
        &norm_beta,
        &gate,
        &receptance,
        &replacement_key,
        &value,
        &bonus,
        &output,
    ]);

    let cube_count = CubeCount::Static(rows as u32, num_heads as u32, 1);
    let cube_dim = CubeDim::new_1d(BLOCK_SIZE);

    unsafe {
        gated_readout_combine_forward_kernel::launch_unchecked::<F, R>(
            &client,
            cube_count,
            cube_dim,
            address_type,
            wkv_output.into_linear_view(),
            norm_gamma.into_linear_view(),
            norm_beta.into_linear_view(),
            gate.into_linear_view_like(&output),
            receptance.into_linear_view(),
            replacement_key.into_linear_view(),
            value.into_linear_view(),
            bonus.into_linear_view(),
            output.clone().into_linear_view(),
            HEAD_SIZE,
            head_size,
            embedded_dim,
            norm_epsilon,
            NUM_WARPS,
        );
    }

    output
}

fn flatten_4d_shape(shape: &Shape) -> [usize; 4] {
    let dims: [usize; 4] = shape.dims();
    [dims[0] * dims[1], dims[2], dims[3], dims[2] * dims[3]]
}

fn max_address_type<R: CubeRuntime>(tensors: &[&CubeTensor<R>]) -> AddressType {
    tensors
        .iter()
        .map(|tensor| tensor.required_address_type())
        .max()
        .unwrap_or_default()
}
