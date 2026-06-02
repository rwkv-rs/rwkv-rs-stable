use burn::tensor::{DType, Shape, ops::FloatTensor};
use burn_cubecl::{
    CubeBackend,
    CubeElement,
    CubeRuntime,
    CubeTuneId,
    FloatElement,
    IntElement,
    cubecl::{
        CubeCount,
        CubeDim,
        prelude::*,
        tune::{AutotuneKey, LocalTuner, Tunable, TunableSet, TuneGroup, anchor, local_tuner},
    },
    element::BoolElement,
    ops::numeric::empty_device,
    tensor::CubeTensor,
};
use serde::{Deserialize, Serialize};

use crate::kernels::train::{
    layout::CubeHardwareFingerprint,
    lm_head_l2wrap_ce::{
        io::LmHeadL2WrapCePrimitiveInputs,
        kernel::{
            LmHeadL2WrapCeInputsLaunch,
            lm_head_l2wrap_ce_forward_finalize_kernel,
            lm_head_l2wrap_ce_forward_row_kernel,
        },
    },
};

const BLOCK_SIZE_CANDIDATES: [usize; 3] = [256, 512, 1024];
const WARP_SIZE: usize = 32;

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
struct LmHeadL2WrapCeForwardAutotuneKey {
    runtime: String,
    dtype: DType,
    num_tokens: usize,
    vocab_size: usize,
    hardware: CubeHardwareFingerprint,
    is_in_place: bool,
    deterministic: bool,
}

impl core::fmt::Display for LmHeadL2WrapCeForwardAutotuneKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{}:{:?}:tokens{}:vocab{}:{}:inplace{}:det{}",
            self.runtime,
            self.dtype,
            self.num_tokens,
            self.vocab_size,
            self.hardware,
            self.is_in_place,
            self.deterministic
        )
    }
}

impl AutotuneKey for LmHeadL2WrapCeForwardAutotuneKey {}

pub(crate) fn fused_lm_head_l2wrap_ce<
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
>(
    inputs: LmHeadL2WrapCePrimitiveInputs<CubeBackend<R, F, I, BT>>,
) -> FloatTensor<CubeBackend<R, F, I, BT>> {
    let LmHeadL2WrapCePrimitiveInputs { logits, targets } = inputs;
    let client = logits.client.clone();

    let key = |(logits, _targets): &(CubeTensor<R>, CubeTensor<R>)| {
        let shape = logits.meta.shape();
        let [batch_size, context_len, vocab_size] = shape.dims();
        let hardware = CubeHardwareFingerprint::from_hardware(&logits.client.properties().hardware);
        let num_tokens = batch_size * context_len;

        LmHeadL2WrapCeForwardAutotuneKey {
            runtime: R::name(&logits.client).to_owned(),
            dtype: logits.dtype,
            num_tokens: anchor(num_tokens, None, Some(1), None),
            vocab_size,
            hardware,
            is_in_place: false,
            deterministic: true,
        }
    };

    let input_gen = |_key: &LmHeadL2WrapCeForwardAutotuneKey,
                     (logits, targets): &(CubeTensor<R>, CubeTensor<R>)| {
        (logits.copy(), targets.copy())
    };

    static TUNER: LocalTuner<LmHeadL2WrapCeForwardAutotuneKey, CubeTuneId> =
        local_tuner!("lm-head-l2wrap-ce-forward");

    let tunables = TUNER.init(move || {
        let block_group = TuneGroup::<LmHeadL2WrapCeForwardAutotuneKey>::new("block_size", |_| 1);
        let mut set = TunableSet::new(key, input_gen);

        for block_size in BLOCK_SIZE_CANDIDATES {
            set = set.with(
                Tunable::new(&format!("block_{block_size}"), move |(logits, targets)| {
                    Ok::<_, String>(lm_head_l2wrap_ce_with_block::<R, F, I, BT>(
                        logits, targets, block_size,
                    ))
                })
                .group(&block_group, move |key| {
                    if block_size as u32 <= key.hardware.max_units_per_cube {
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
        &CubeTuneId::new(&client, &logits.device),
        &client,
        tunables,
        (logits, targets),
    )
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
    use crate::kernels::train::lm_head_l2wrap_ce::LmHeadL2WrapCeBackend;

    impl<B: FusionBackend + LmHeadL2WrapCeBackend> LmHeadL2WrapCeBackend for Fusion<B> {
        fn fused_lm_head_l2wrap_ce(
            inputs: LmHeadL2WrapCePrimitiveInputs<Self>,
        ) -> FloatTensor<Self> {
            let LmHeadL2WrapCePrimitiveInputs { logits, targets } = inputs;
            let client = logits.client.clone();

            #[derive(Clone, Debug)]
            struct LmHeadL2WrapCeOp<B1> {
                desc: CustomOpIr,
                _backend: core::marker::PhantomData<B1>,
            }

            impl<B1: FusionBackend + LmHeadL2WrapCeBackend> Operation<B1::FusionRuntime>
                for LmHeadL2WrapCeOp<B1>
            {
                fn execute(
                    &self,
                    handles: &mut HandleContainer<
                        <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                    >,
                ) {
                    let ([logits, targets], [loss_out]) = self.desc.as_fixed();
                    let loss = B1::fused_lm_head_l2wrap_ce(LmHeadL2WrapCePrimitiveInputs {
                        logits: handles.get_float_tensor::<B1>(logits),
                        targets: handles.get_int_tensor::<B1>(targets),
                    });

                    handles.register_float_tensor::<B1>(&loss_out.id, loss);
                }
            }

            let mut streams = OperationStreams::default();
            streams.tensor(&logits);
            streams.tensor(&targets);

            let output_desc = [TensorIr::uninit(
                client.create_empty_handle(),
                Shape::new([1]),
                B::FloatElem::dtype(),
            )];
            let desc = CustomOpIr::new(
                "fused_lm_head_l2wrap_ce",
                &[logits.into_ir(), targets.into_ir()],
                &output_desc,
            );
            let op = LmHeadL2WrapCeOp::<B> {
                desc,
                _backend: core::marker::PhantomData,
            };

            client
                .register(streams, OperationIr::Custom(op.desc.clone()), op)
                .pop()
                .expect("missing fused_lm_head_l2wrap_ce output")
        }
    }
}

fn lm_head_l2wrap_ce_with_block<R, F, I, BT>(
    logits: CubeTensor<R>,
    targets: CubeTensor<R>,
    block_size: usize,
) -> FloatTensor<CubeBackend<R, F, I, BT>>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
{
    let shape = logits.meta.shape().clone();
    let [batch_size, context_len, vocab_size] = shape.dims();
    let num_tokens = batch_size * context_len;
    let client = logits.client.clone();
    let device = logits.device.clone();
    let row_losses = empty_device::<R, F>(client.clone(), device.clone(), Shape::new([num_tokens]));
    let loss = empty_device::<R, F>(client.clone(), device, Shape::new([1]));
    let num_warps = block_size / WARP_SIZE;
    let row_address_type = max_address_type(&[&logits, &targets, &row_losses]);

    unsafe {
        lm_head_l2wrap_ce_forward_row_kernel::launch_unchecked::<F, I, R>(
            &client,
            CubeCount::Static(num_tokens as u32, 1, 1),
            CubeDim::new_1d(block_size as u32),
            row_address_type,
            LmHeadL2WrapCeInputsLaunch::new(
                logits.clone().into_linear_view(),
                targets.clone().into_linear_view(),
            ),
            row_losses.clone().into_linear_view(),
            vocab_size,
            num_tokens,
            block_size,
            num_warps,
        );
    }

    let finalize_address_type = max_address_type(&[&row_losses, &loss]);
    // SAFETY: The finalize launch writes the single allocated loss element and only scans the
    // allocated row-loss buffer.
    unsafe {
        lm_head_l2wrap_ce_forward_finalize_kernel::launch_unchecked::<F, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(256),
            finalize_address_type,
            row_losses.into_linear_view(),
            loss.clone().into_linear_view(),
            num_tokens,
            num_tokens,
            8,
        );
    }

    loss
}

pub(crate) fn max_address_type<R: CubeRuntime>(tensors: &[&CubeTensor<R>]) -> AddressType {
    tensors
        .iter()
        .map(|tensor| tensor.required_address_type())
        .max()
        .unwrap_or_default()
}
