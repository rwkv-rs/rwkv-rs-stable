use burn_cubecl::{
    CubeBackend,
    CubeElement,
    CubeRuntime,
    FloatElement,
    IntElement,
    cubecl::{CubeCount, calculate_cube_count_elemwise, prelude::*},
    element::BoolElement,
    ops::numeric::empty_device,
    tensor::CubeTensor,
};

use crate::kernels::train::time_mixer::key_prepare::{
    io::{KeyPrepareForwardPrimitiveInputs, KeyPrepareForwardPrimitiveOutput},
    kernel::{
        KeyPrepareForwardInputsLaunch,
        KeyPrepareForwardOutputsLaunch,
        key_prepare_forward_64_kernel,
        key_prepare_forward_kernel,
    },
};

const HEAD64_WARPS_PER_CUBE: usize = 4;

pub(crate) fn fused_key_prepare<
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
>(
    inputs: KeyPrepareForwardPrimitiveInputs<CubeBackend<R, F, I, BT>>,
) -> KeyPrepareForwardPrimitiveOutput<CubeBackend<R, F, I, BT>> {
    let KeyPrepareForwardPrimitiveInputs {
        key,
        key_removal,
        learning_rate,
        key_replacement,
        head_size,
    } = inputs;

    key_prepare::<R, F, I, BT>(key, key_removal, learning_rate, key_replacement, head_size)
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
    use crate::kernels::train::time_mixer::key_prepare::KeyPrepareBackend;

    impl<B: FusionBackend + KeyPrepareBackend> KeyPrepareBackend for Fusion<B> {
        fn fused_key_prepare(
            inputs: KeyPrepareForwardPrimitiveInputs<Self>,
        ) -> KeyPrepareForwardPrimitiveOutput<Self> {
            let KeyPrepareForwardPrimitiveInputs {
                key,
                key_removal,
                learning_rate,
                key_replacement,
                head_size,
            } = inputs;
            let client = key.client.clone();
            let [batch_size, context_len, embedded_dim] = key.shape.dims();

            #[derive(Clone, Debug)]
            struct KeyPrepareOp<B1> {
                desc: CustomOpIr,
                head_size: usize,
                _backend: core::marker::PhantomData<B1>,
            }

            impl<B1: FusionBackend + KeyPrepareBackend> Operation<B1::FusionRuntime> for KeyPrepareOp<B1> {
                fn execute(
                    &self,
                    handles: &mut HandleContainer<
                        <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                    >,
                ) {
                    let (
                        [key, key_removal, learning_rate, key_replacement],
                        [
                            replacement_key_out,
                            removal_key_normalized_out,
                            replacement_out,
                        ],
                    ) = self.desc.as_fixed();

                    let output = B1::fused_key_prepare(KeyPrepareForwardPrimitiveInputs {
                        key: handles.get_float_tensor::<B1>(key),
                        key_removal: handles.get_float_tensor::<B1>(key_removal),
                        learning_rate: handles.get_float_tensor::<B1>(learning_rate),
                        key_replacement: handles.get_float_tensor::<B1>(key_replacement),
                        head_size: self.head_size,
                    });

                    handles.register_float_tensor::<B1>(
                        &replacement_key_out.id,
                        output.replacement_key,
                    );
                    handles.register_float_tensor::<B1>(
                        &removal_key_normalized_out.id,
                        output.removal_key_normalized,
                    );
                    handles.register_float_tensor::<B1>(&replacement_out.id, output.replacement);
                }
            }

            let mut streams = OperationStreams::default();
            streams.tensor(&key);
            streams.tensor(&key_removal);
            streams.tensor(&learning_rate);
            streams.tensor(&key_replacement);

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
            ];

            let desc = CustomOpIr::new(
                "fused_key_prepare",
                &[
                    key.into_ir(),
                    key_removal.into_ir(),
                    learning_rate.into_ir(),
                    key_replacement.into_ir(),
                ],
                &output_desc,
            );

            let op = KeyPrepareOp::<B> {
                desc,
                head_size,
                _backend: core::marker::PhantomData,
            };

            let mut outputs = client.register(streams, OperationIr::Custom(op.desc.clone()), op);
            let replacement = outputs.pop().expect("missing replacement");
            let removal_key_normalized = outputs.pop().expect("missing removal_key_normalized");
            let replacement_key = outputs.pop().expect("missing replacement_key");

            KeyPrepareForwardPrimitiveOutput {
                replacement_key,
                removal_key_normalized,
                replacement,
            }
        }
    }
}

fn key_prepare<R, F, I, BT>(
    key: CubeTensor<R>,
    key_removal: CubeTensor<R>,
    learning_rate: CubeTensor<R>,
    key_replacement: CubeTensor<R>,
    head_size: usize,
) -> KeyPrepareForwardPrimitiveOutput<CubeBackend<R, F, I, BT>>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
{
    let shape = key.meta.shape().clone();
    let client = key.client.clone();
    let replacement_key = empty_device::<R, F>(client.clone(), key.device.clone(), shape.clone());
    let removal_key_normalized =
        empty_device::<R, F>(client.clone(), key.device.clone(), shape.clone());
    let replacement = empty_device::<R, F>(client.clone(), key.device.clone(), shape.clone());

    if shape.num_elements() == 0 {
        return KeyPrepareForwardPrimitiveOutput {
            replacement_key,
            removal_key_normalized,
            replacement,
        };
    }

    if head_size == 64 {
        let num_heads = shape[2] / head_size;
        let bth_len = shape[0] * shape[1] * num_heads;
        let cubes = bth_len.div_ceil(HEAD64_WARPS_PER_CUBE) as u32;
        let address_type = max_address_type(&[
            &key,
            &key_removal,
            &learning_rate,
            &key_replacement,
            &replacement_key,
            &removal_key_normalized,
            &replacement,
        ]);

        // The RWKV7 trace contract uses `head_size = 64`. Match the CUDA reference layout: one
        // warp owns one `[batch, time, head]` vector, each lane computes two BF16 values, and the
        // head norm is reduced once across the warp instead of recomputed by every element.
        // SAFETY: The public input contract checks shape/dtype/device and that `head_size` divides
        // the embedded dimension. The specialized path writes exactly the same output shapes as the
        // generic fallback.
        unsafe {
            key_prepare_forward_64_kernel::launch_unchecked::<F, R>(
                &client,
                CubeCount::Static(cubes, 1, 1),
                CubeDim::new_1d((HEAD64_WARPS_PER_CUBE * 32) as u32),
                address_type,
                KeyPrepareForwardInputsLaunch::new(
                    key.into_linear_view_like(&replacement_key),
                    key_removal.into_linear_view(),
                    learning_rate.into_linear_view_like(&replacement_key),
                    key_replacement.into_linear_view(),
                ),
                KeyPrepareForwardOutputsLaunch::new(
                    replacement_key.clone().into_linear_view(),
                    removal_key_normalized.clone().into_linear_view(),
                    replacement.clone().into_linear_view(),
                ),
                num_heads,
                bth_len,
                HEAD64_WARPS_PER_CUBE,
            );
        }

        return KeyPrepareForwardPrimitiveOutput {
            replacement_key,
            removal_key_normalized,
            replacement,
        };
    }

    let working_units = shape.num_elements();
    let cube_dim = CubeDim::new(&client, working_units);
    let cube_count = calculate_cube_count_elemwise(&client, working_units, cube_dim);
    let address_type = max_address_type(&[
        &key,
        &key_removal,
        &learning_rate,
        &key_replacement,
        &replacement_key,
        &removal_key_normalized,
        &replacement,
    ]);

    let head_mask = if head_size > 1 && head_size.is_power_of_two() {
        head_size - 1
    } else {
        0
    };

    // Each work unit owns one `[batch_size, context_len, embedded_dim]` element. It recomputes the
    // L2 norm for the element's head and then writes all three RWKV-LM v7 key-prepare outputs.
    // This prioritizes the exact fused contract and avoids intermediate tensors; a later tuned
    // version can replace the per-element head loop with a shared or warp reduction. RWKV-LM trace
    // shapes use `head_size=64`, so the kernel receives a mask to avoid repeated modulo on the hot
    // power-of-two lane path while retaining the checked fallback for generic callers.
    // SAFETY: The public contract checks dtype/device/shape and that `head_size` divides
    // `embedded_dim`; primitive dispatch checks contiguity, and all outputs use the input shape.
    unsafe {
        key_prepare_forward_kernel::launch_unchecked::<F, R>(
            &client,
            cube_count,
            cube_dim,
            address_type,
            KeyPrepareForwardInputsLaunch::new(
                key.into_linear_view_like(&replacement_key),
                key_removal.into_linear_view(),
                learning_rate.into_linear_view_like(&replacement_key),
                key_replacement.into_linear_view(),
            ),
            KeyPrepareForwardOutputsLaunch::new(
                replacement_key.clone().into_linear_view(),
                removal_key_normalized.clone().into_linear_view(),
                replacement.clone().into_linear_view(),
            ),
            shape[2],
            head_size,
            head_mask,
        );
    }

    KeyPrepareForwardPrimitiveOutput {
        replacement_key,
        removal_key_normalized,
        replacement,
    }
}

fn max_address_type<R: CubeRuntime>(tensors: &[&CubeTensor<R>]) -> AddressType {
    tensors
        .iter()
        .map(|tensor| tensor.required_address_type())
        .max()
        .unwrap_or_default()
}
