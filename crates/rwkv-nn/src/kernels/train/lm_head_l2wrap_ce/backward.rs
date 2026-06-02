use burn::{
    backend::autodiff::{
        Autodiff,
        checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
        grads::Gradients,
        ops::{Backward, Ops, OpsKind},
    },
    tensor::{
        DType,
        ops::{FloatTensor, IntTensor},
    },
};
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
        tune::{AutotuneKey, LocalTuner, Tunable, TunableSet, TuneGroup, anchor, local_tuner},
    },
    element::BoolElement,
    ops::numeric::empty_device,
    tensor::CubeTensor,
};
use serde::{Deserialize, Serialize};

use crate::kernels::train::{
    layout::{CubeHardwareFingerprint, assert_linear_readable},
    lm_head_l2wrap_ce::{
        L2WRAP_FACTOR,
        LmHeadL2WrapCeBackend,
        forward,
        io::LmHeadL2WrapCePrimitiveInputs,
        kernel::{LmHeadL2WrapCeInputsLaunch, lm_head_l2wrap_ce_backward_kernel},
    },
};

const BLOCK_SIZE_CANDIDATES: [usize; 3] = [256, 512, 1024];
const WARP_SIZE: usize = 32;

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
struct LmHeadL2WrapCeBackwardAutotuneKey {
    runtime: String,
    dtype: DType,
    num_tokens: usize,
    vocab_size: usize,
    hardware: CubeHardwareFingerprint,
    is_in_place: bool,
    deterministic: bool,
}

impl core::fmt::Display for LmHeadL2WrapCeBackwardAutotuneKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{}:{:?}:tokens{}:vocab{}:hw{}:inplace{}:det{}",
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

impl AutotuneKey for LmHeadL2WrapCeBackwardAutotuneKey {}

impl<R, F, I, BT, C> LmHeadL2WrapCeBackend for Autodiff<CubeBackend<R, F, I, BT>, C>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
    C: CheckpointStrategy,
{
    fn fused_lm_head_l2wrap_ce(inputs: LmHeadL2WrapCePrimitiveInputs<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct FusedLmHeadL2WrapCeBackward;

        impl<R, F, I, BT> Backward<CubeBackend<R, F, I, BT>, 1> for FusedLmHeadL2WrapCeBackward
        where
            R: CubeRuntime,
            F: FloatElement + CubeElement,
            I: IntElement,
            BT: BoolElement,
        {
            type State = LmHeadL2WrapCeBackwardState<CubeBackend<R, F, I, BT>>;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let [node_logits] = ops.parents;
                let output_grad = grads.consume::<CubeBackend<R, F, I, BT>>(&ops.node);
                let LmHeadL2WrapCeBackwardState { logits, targets } = ops.state;
                let logits_grad =
                    lm_head_l2wrap_ce_backward::<R, F, I, BT>(output_grad, logits, targets);

                if let Some(node) = node_logits {
                    grads.register::<CubeBackend<R, F, I, BT>>(node.id, logits_grad);
                }
            }
        }

        let LmHeadL2WrapCePrimitiveInputs { logits, targets } = inputs;
        assert_linear_readable("logits", &logits.primitive);
        assert_linear_readable("targets", &targets);
        let logits_primitive = logits.primitive.clone();
        let targets_primitive = targets.clone();
        let output =
            forward::fused_lm_head_l2wrap_ce::<R, F, I, BT>(LmHeadL2WrapCePrimitiveInputs {
                logits: logits_primitive.clone(),
                targets: targets_primitive.clone(),
            });

        match FusedLmHeadL2WrapCeBackward
            .prepare::<C>([logits.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(
                LmHeadL2WrapCeBackwardState {
                    logits: logits_primitive,
                    targets: targets_primitive,
                },
                output,
            ),
            OpsKind::UnTracked(prep) => prep.finish(output),
        }
    }
}

#[derive(Clone, Debug)]
struct LmHeadL2WrapCeBackwardState<B: burn::tensor::backend::Backend> {
    logits: FloatTensor<B>,
    targets: IntTensor<B>,
}

fn lm_head_l2wrap_ce_backward<R, F, I, BT>(
    output_grad: CubeTensor<R>,
    logits: CubeTensor<R>,
    targets: CubeTensor<R>,
) -> FloatTensor<CubeBackend<R, F, I, BT>>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
{
    assert_linear_readable("output_grad", &output_grad);
    assert_linear_readable("logits", &logits);
    assert_linear_readable("targets", &targets);

    let client = logits.client.clone();

    let key = |(_output_grad, logits, _targets): &(CubeTensor<R>, CubeTensor<R>, CubeTensor<R>)| {
        let shape = logits.meta.shape();
        let [batch_size, context_len, vocab_size] = shape.dims();
        let hardware = CubeHardwareFingerprint::from_hardware(&logits.client.properties().hardware);

        LmHeadL2WrapCeBackwardAutotuneKey {
            runtime: R::name(&logits.client).to_owned(),
            dtype: logits.dtype,
            num_tokens: anchor(batch_size * context_len, None, Some(1), None),
            vocab_size,
            hardware,
            is_in_place: false,
            deterministic: true,
        }
    };

    let input_gen =
        |_key: &LmHeadL2WrapCeBackwardAutotuneKey,
         (output_grad, logits, targets): &(CubeTensor<R>, CubeTensor<R>, CubeTensor<R>)| {
            (output_grad.copy(), logits.copy(), targets.copy())
        };

    static TUNER: LocalTuner<LmHeadL2WrapCeBackwardAutotuneKey, CubeTuneId> =
        local_tuner!("lm-head-l2wrap-ce-backward");

    let tunables = TUNER.init(move || {
        let block_group = TuneGroup::<LmHeadL2WrapCeBackwardAutotuneKey>::new("block_size", |_| 1);
        let mut set = TunableSet::new(key, input_gen);

        for block_size in BLOCK_SIZE_CANDIDATES {
            set = set.with(
                Tunable::new(
                    &format!("block_{block_size}"),
                    move |(output_grad, logits, targets)| {
                        Ok::<_, String>(lm_head_l2wrap_ce_backward_with_block::<R, F, I, BT>(
                            output_grad,
                            logits,
                            targets,
                            block_size,
                        ))
                    },
                )
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
        (output_grad, logits, targets),
    )
}

fn lm_head_l2wrap_ce_backward_with_block<R, F, I, BT>(
    output_grad: CubeTensor<R>,
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
    assert_linear_readable("output_grad", &output_grad);
    assert_linear_readable("logits", &logits);
    assert_linear_readable("targets", &targets);

    let shape = logits.meta.shape().clone();
    let vocab_size = shape[2];
    let num_tokens = shape[0] * shape[1];
    let client = logits.client.clone();
    let logits_grad = empty_device::<R, F>(client.clone(), logits.device.clone(), shape);
    let address_type = forward::max_address_type(&[&logits, &targets, &output_grad, &logits_grad]);
    let num_warps = block_size / WARP_SIZE;

    // SAFETY: The forward public contract checked shape/device; this launch guards the flattened
    // element range against the allocated logits gradient.
    unsafe {
        lm_head_l2wrap_ce_backward_kernel::launch_unchecked::<F, I, R>(
            &client,
            CubeCount::Static(num_tokens as u32, 1, 1),
            CubeDim::new_1d(block_size as u32),
            address_type,
            LmHeadL2WrapCeInputsLaunch::new(logits.into_linear_view(), targets.into_linear_view()),
            output_grad.into_linear_view(),
            logits_grad.clone().into_linear_view(),
            vocab_size,
            num_tokens,
            L2WRAP_FACTOR as f32,
            block_size,
            num_warps,
        );
    }

    logits_grad
}
