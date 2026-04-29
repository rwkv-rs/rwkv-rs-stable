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

use crate::kernels::train::time_mixer::wkv7::{
    io::{
        Wkv7PretrainForwardPrimitiveInputs,
        Wkv7StatepassForwardPrimitiveInputs,
        Wkv7StatepassForwardPrimitiveOutput,
        Wkv7StatetuneForwardPrimitiveInputs,
    },
    kernel::{Wkv7ForwardInputsLaunch, wkv7_pretrain_forward_kernel, wkv7_state_forward_kernel},
};

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
struct Wkv7ForwardAutotuneKey {
    op: Wkv7ForwardOp,
    dtype: DType,
    batch_heads: usize,
    context_len: usize,
    head_size: usize,
    chunk_len: usize,
}

impl core::fmt::Display for Wkv7ForwardAutotuneKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{:?}:{:?}:{}:{}:{}:{}",
            self.op, self.dtype, self.batch_heads, self.context_len, self.head_size, self.chunk_len
        )
    }
}

impl AutotuneKey for Wkv7ForwardAutotuneKey {}

#[derive(Hash, Eq, PartialEq, Debug, Clone, Copy, Serialize, Deserialize)]
enum Wkv7ForwardOp {
    Pretrain,
}

pub(crate) fn fused_wkv7_pretrain<
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
>(
    inputs: Wkv7PretrainForwardPrimitiveInputs<CubeBackend<R, F, I, BT>>,
) -> FloatTensor<CubeBackend<R, F, I, BT>> {
    let Wkv7PretrainForwardPrimitiveInputs {
        receptance,
        weight_decay,
        replacement_key,
        value,
        removal_key_normalized,
        replacement,
        chunk_len,
    } = inputs;
    let client = receptance.client.clone();

    let key_chunk_len = chunk_len;
    let key = move |receptance: &CubeTensor<R>,
                    _weight_decay: &CubeTensor<R>,
                    _replacement_key: &CubeTensor<R>,
                    _value: &CubeTensor<R>,
                    _removal_key_normalized: &CubeTensor<R>,
                    _replacement: &CubeTensor<R>| {
        let shape = receptance.meta.shape();
        Wkv7ForwardAutotuneKey {
            op: Wkv7ForwardOp::Pretrain,
            dtype: receptance.dtype,
            batch_heads: anchor(shape[0] * shape[2], None, Some(1), None),
            context_len: shape[1],
            head_size: shape[3],
            chunk_len: key_chunk_len,
        }
    };
    let input_gen = |_key: &Wkv7ForwardAutotuneKey,
                     receptance: &CubeTensor<R>,
                     weight_decay: &CubeTensor<R>,
                     replacement_key: &CubeTensor<R>,
                     value: &CubeTensor<R>,
                     removal_key_normalized: &CubeTensor<R>,
                     replacement: &CubeTensor<R>| {
        (
            receptance.copy(),
            weight_decay.copy(),
            replacement_key.copy(),
            value.copy(),
            removal_key_normalized.copy(),
            replacement.copy(),
        )
    };

    static TUNER: LocalTuner<Wkv7ForwardAutotuneKey, CubeTuneId> =
        local_tuner!("rwkv7-wkv-pretrain-forward");

    let tunables = TUNER.init(move || {
        let variant_group = TuneGroup::<Wkv7ForwardAutotuneKey>::new("variant", |_| 1);
        let mut set = TunableSet::new(key, input_gen);

        for variant in [
            Wkv7PretrainForwardVariant::V1,
            Wkv7PretrainForwardVariant::V3Preload,
        ] {
            set = set.with(
                Tunable::new(
                    format!("{variant:?}"),
                    (move |receptance: CubeTensor<R>,
                           weight_decay: CubeTensor<R>,
                           replacement_key: CubeTensor<R>,
                           value: CubeTensor<R>,
                           removal_key_normalized: CubeTensor<R>,
                           replacement: CubeTensor<R>| {
                        wkv7_pretrain::<R, F>(
                            Wkv7PretrainLaunchInputs {
                                receptance,
                                weight_decay,
                                replacement_key,
                                value,
                                removal_key_normalized,
                                replacement,
                                chunk_len,
                            },
                            variant,
                        )
                        .output
                    })
                    .ok(),
                )
                .group(&variant_group, |_| 1),
            );
        }

        set
    });

    TUNER.execute(
        &CubeTuneId::new(&receptance.client, &receptance.device),
        &client,
        tunables,
        (
            receptance,
            weight_decay,
            replacement_key,
            value,
            removal_key_normalized,
            replacement,
        ),
    )
}

#[allow(dead_code)]
pub(crate) fn fused_wkv7_pretrain_with_saved<
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
>(
    inputs: Wkv7PretrainForwardPrimitiveInputs<CubeBackend<R, F, I, BT>>,
) -> Wkv7PretrainForwardSaved<R> {
    wkv7_pretrain::<R, F>(
        Wkv7PretrainLaunchInputs::from_sequence(inputs),
        Wkv7PretrainForwardVariant::V1,
    )
}

pub(crate) fn fused_wkv7_statetune<
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
>(
    inputs: Wkv7StatetuneForwardPrimitiveInputs<CubeBackend<R, F, I, BT>>,
) -> FloatTensor<CubeBackend<R, F, I, BT>> {
    let Wkv7StatetuneForwardPrimitiveInputs {
        initial_state,
        sequence,
    } = inputs;
    wkv7_state::<R, F, I, BT>(
        initial_state,
        Wkv7PretrainLaunchInputs::from_sequence(sequence),
        false,
    )
    .output
}

#[allow(dead_code)]
pub(crate) fn fused_wkv7_statetune_with_saved<
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
>(
    inputs: Wkv7StatetuneForwardPrimitiveInputs<CubeBackend<R, F, I, BT>>,
) -> Wkv7StateForwardSaved<R> {
    let Wkv7StatetuneForwardPrimitiveInputs {
        initial_state,
        sequence,
    } = inputs;
    wkv7_state::<R, F, I, BT>(
        initial_state,
        Wkv7PretrainLaunchInputs::from_sequence(sequence),
        true,
    )
}

pub(crate) fn fused_wkv7_statepass<
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
>(
    inputs: Wkv7StatepassForwardPrimitiveInputs<CubeBackend<R, F, I, BT>>,
) -> Wkv7StatepassForwardPrimitiveOutput<CubeBackend<R, F, I, BT>> {
    let Wkv7StatepassForwardPrimitiveInputs {
        initial_state,
        sequence,
    } = inputs;
    let output = wkv7_state::<R, F, I, BT>(
        initial_state,
        Wkv7PretrainLaunchInputs::from_sequence(sequence),
        true,
    );

    Wkv7StatepassForwardPrimitiveOutput {
        output: output.output,
        next_state: output.next_state,
    }
}

#[allow(dead_code)]
pub(crate) fn fused_wkv7_statepass_with_saved<
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
>(
    inputs: Wkv7StatepassForwardPrimitiveInputs<CubeBackend<R, F, I, BT>>,
) -> Wkv7StateForwardSaved<R> {
    let Wkv7StatepassForwardPrimitiveInputs {
        initial_state,
        sequence,
    } = inputs;
    wkv7_state::<R, F, I, BT>(
        initial_state,
        Wkv7PretrainLaunchInputs::from_sequence(sequence),
        true,
    )
}

#[allow(dead_code)]
pub(crate) struct Wkv7PretrainForwardSaved<R: CubeRuntime> {
    pub(crate) output: CubeTensor<R>,
    pub(crate) snapshots: CubeTensor<R>,
    pub(crate) state_replacement: CubeTensor<R>,
}

#[allow(dead_code)]
pub(crate) struct Wkv7StateForwardSaved<R: CubeRuntime> {
    pub(crate) output: CubeTensor<R>,
    pub(crate) next_state: CubeTensor<R>,
    pub(crate) snapshots: CubeTensor<R>,
    pub(crate) state_replacement: CubeTensor<R>,
}

#[derive(Clone, Copy, Debug)]
enum Wkv7PretrainForwardVariant {
    V1,
    V3Preload,
}

struct Wkv7PretrainLaunchInputs<R: CubeRuntime> {
    receptance: CubeTensor<R>,
    weight_decay: CubeTensor<R>,
    replacement_key: CubeTensor<R>,
    value: CubeTensor<R>,
    removal_key_normalized: CubeTensor<R>,
    replacement: CubeTensor<R>,
    chunk_len: usize,
}

impl<R: CubeRuntime> Wkv7PretrainLaunchInputs<R> {
    fn from_sequence(
        sequence: Wkv7PretrainForwardPrimitiveInputs<
            CubeBackend<R, impl FloatElement + CubeElement, impl IntElement, impl BoolElement>,
        >,
    ) -> Self {
        let Wkv7PretrainForwardPrimitiveInputs {
            receptance,
            weight_decay,
            replacement_key,
            value,
            removal_key_normalized,
            replacement,
            chunk_len,
        } = sequence;

        Self {
            receptance,
            weight_decay,
            replacement_key,
            value,
            removal_key_normalized,
            replacement,
            chunk_len,
        }
    }
}

fn wkv7_pretrain<R, F>(
    inputs: Wkv7PretrainLaunchInputs<R>,
    _variant: Wkv7PretrainForwardVariant,
) -> Wkv7PretrainForwardSaved<R>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
{
    let Wkv7PretrainLaunchInputs {
        receptance,
        weight_decay,
        replacement_key,
        value,
        removal_key_normalized,
        replacement,
        chunk_len,
    } = inputs;
    let shape = receptance.meta.shape().clone();
    let client = receptance.client.clone();
    let device = receptance.device.clone();
    let output = empty_device::<R, F>(client.clone(), device.clone(), shape.clone());
    let dtype = receptance.dtype;
    let snapshots = empty_device::<R, F>(
        client.clone(),
        device.clone(),
        Shape::new([shape[0], shape[2], shape[1] / chunk_len, shape[3], shape[3]]),
    );
    let state_replacement = empty_device::<R, F>(client.clone(), device, shape.clone());

    if shape.num_elements() == 0 {
        return Wkv7PretrainForwardSaved {
            output,
            snapshots,
            state_replacement,
        };
    }

    let cube_dim = CubeDim::new_1d(shape[3] as u32);
    let cube_count = CubeCount::Static(shape[2] as u32, shape[0] as u32, 1);
    let address_type = max_address_type(&[
        &receptance,
        &weight_decay,
        &replacement_key,
        &value,
        &removal_key_normalized,
        &replacement,
        &output,
        &snapshots,
        &state_replacement,
    ]);

    // One cube owns one `[batch_size, num_heads]` pair. One unit owns one output/state row and
    // scans the `head_size` columns sequentially, matching the CUDA recurrence dependency shape.
    // The v1 and preload autotune candidates currently share this conservative CubeCL launch; the
    // candidate names preserve the intended performance split for later shared-memory preloading.
    // SAFETY: The public input contract checks shape/dtype/device/chunk length, and primitive
    // dispatch checks contiguity. The launch grid covers every batch/head pair and every state row.
    unsafe {
        wkv7_pretrain_forward_kernel::launch_unchecked::<R>(
            &client,
            cube_count,
            cube_dim,
            address_type,
            Wkv7ForwardInputsLaunch::new(
                receptance.into_linear_view(),
                weight_decay.into_linear_view(),
                replacement_key.into_linear_view(),
                value.into_linear_view(),
                removal_key_normalized.into_linear_view(),
                replacement.into_linear_view(),
            ),
            output.clone().into_linear_view(),
            snapshots.clone().into_linear_view(),
            state_replacement.clone().into_linear_view(),
            shape[1],
            shape[2],
            shape[3],
            chunk_len,
            dtype.into(),
        );
    }

    Wkv7PretrainForwardSaved {
        output,
        snapshots,
        state_replacement,
    }
}

fn wkv7_state<R, F, I, BT>(
    initial_state: CubeTensor<R>,
    inputs: Wkv7PretrainLaunchInputs<R>,
    need_next_state: bool,
) -> Wkv7StateForwardSaved<R>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
{
    let Wkv7PretrainLaunchInputs {
        receptance,
        weight_decay,
        replacement_key,
        value,
        removal_key_normalized,
        replacement,
        chunk_len,
    } = inputs;
    let shape = receptance.meta.shape().clone();
    let client = receptance.client.clone();
    let device = receptance.device.clone();
    let output = empty_device::<R, F>(client.clone(), device.clone(), shape.clone());
    let dtype = receptance.dtype;
    let next_state = empty_device::<R, F>(
        client.clone(),
        device.clone(),
        Shape::new([shape[0], shape[2], shape[3], shape[3]]),
    );
    let snapshots = empty_device::<R, F>(
        client.clone(),
        device.clone(),
        Shape::new([shape[0], shape[2], shape[1] / chunk_len, shape[3], shape[3]]),
    );
    let state_replacement = empty_device::<R, F>(client.clone(), device, shape.clone());

    if shape.num_elements() > 0 {
        let cube_dim = CubeDim::new_1d(shape[3] as u32);
        let cube_count = CubeCount::Static(shape[2] as u32, shape[0] as u32, 1);
        let address_type = max_address_type(&[
            &initial_state,
            &receptance,
            &weight_decay,
            &replacement_key,
            &value,
            &removal_key_normalized,
            &replacement,
            &output,
            &next_state,
            &snapshots,
            &state_replacement,
        ]);

        // SAFETY: Same sequence contract as pretrain plus the checked initial-state shape.
        unsafe {
            wkv7_state_forward_kernel::launch_unchecked::<R>(
                &client,
                cube_count,
                cube_dim,
                address_type,
                initial_state.clone().into_linear_view(),
                Wkv7ForwardInputsLaunch::new(
                    receptance.into_linear_view(),
                    weight_decay.into_linear_view(),
                    replacement_key.into_linear_view(),
                    value.into_linear_view(),
                    removal_key_normalized.into_linear_view(),
                    replacement.into_linear_view(),
                ),
                output.clone().into_linear_view(),
                next_state.clone().into_linear_view(),
                snapshots.clone().into_linear_view(),
                state_replacement.clone().into_linear_view(),
                shape[1],
                shape[2],
                shape[3],
                chunk_len,
                dtype.into(),
            );
        }
    }

    Wkv7StateForwardSaved {
        output,
        next_state: if need_next_state {
            next_state
        } else {
            initial_state
        },
        snapshots,
        state_replacement,
    }
}

fn max_address_type<R: CubeRuntime>(tensors: &[&CubeTensor<R>]) -> AddressType {
    tensors
        .iter()
        .map(|tensor| tensor.required_address_type())
        .max()
        .unwrap_or_default()
}
