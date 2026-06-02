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
    time_mixer::wkv7::{
        io::{
            Wkv7PretrainForwardPrimitiveInputs,
            Wkv7StatepassForwardPrimitiveInputs,
            Wkv7StatepassForwardPrimitiveOutput,
            Wkv7StatetuneForwardPrimitiveInputs,
        },
        kernel::{
            Wkv7ForwardInputsLaunch,
            wkv7_pretrain_forward_kernel,
            wkv7_pretrain_forward_output_kernel,
            wkv7_state_forward_kernel,
        },
    },
};

const PRETRAIN_OUTPUT_ROW_TILE_CANDIDATES: [usize; 3] = [16, 32, 64];

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
struct Wkv7PretrainOutputAutotuneKey {
    runtime: String,
    dtype: DType,
    batch_size: usize,
    context_len: usize,
    rows: usize,
    d_model: usize,
    num_heads: usize,
    head_size: usize,
    chunk_len: usize,
    hardware: CubeHardwareFingerprint,
    is_in_place: bool,
    deterministic: bool,
}

impl core::fmt::Display for Wkv7PretrainOutputAutotuneKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{}:{:?}:b{}:t{}:r{}:d{}:h{}:hs{}:chunk{}:{}:inplace{}:det{}",
            self.runtime,
            self.dtype,
            self.batch_size,
            self.context_len,
            self.rows,
            self.d_model,
            self.num_heads,
            self.head_size,
            self.chunk_len,
            self.hardware,
            self.is_in_place,
            self.deterministic
        )
    }
}

impl AutotuneKey for Wkv7PretrainOutputAutotuneKey {}

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
    wkv7_pretrain_output::<R, F>(Wkv7PretrainLaunchInputs {
        receptance,
        weight_decay,
        replacement_key,
        value,
        removal_key_normalized,
        replacement,
        chunk_len,
    })
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
    wkv7_pretrain::<R, F>(Wkv7PretrainLaunchInputs::from_sequence(inputs))
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

fn wkv7_pretrain_output<R, F>(inputs: Wkv7PretrainLaunchInputs<R>) -> CubeTensor<R>
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

    let client = receptance.client.clone();
    let key = move |(
        receptance,
        _weight_decay,
        _replacement_key,
        _value,
        _removal_key_normalized,
        _replacement,
    ): &(
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
    )| {
        let shape = receptance.meta.shape();
        let hardware =
            CubeHardwareFingerprint::from_hardware(&receptance.client.properties().hardware);
        let batch_size = shape[0];
        let context_len = shape[1];
        let num_heads = shape[2];
        let head_size = shape[3];

        Wkv7PretrainOutputAutotuneKey {
            runtime: R::name(&receptance.client).to_owned(),
            dtype: receptance.dtype,
            batch_size,
            context_len,
            rows: anchor(batch_size * context_len, None, Some(1), None),
            d_model: num_heads * head_size,
            num_heads,
            head_size,
            chunk_len,
            hardware,
            is_in_place: false,
            deterministic: true,
        }
    };

    let input_gen = |_key: &Wkv7PretrainOutputAutotuneKey,
                     (
        receptance,
        weight_decay,
        replacement_key,
        value,
        removal_key_normalized,
        replacement,
    ): &(
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
        CubeTensor<R>,
    )| {
        (
            receptance.copy(),
            weight_decay.copy(),
            replacement_key.copy(),
            value.copy(),
            removal_key_normalized.copy(),
            replacement.copy(),
        )
    };

    static TUNER: LocalTuner<Wkv7PretrainOutputAutotuneKey, CubeTuneId> =
        local_tuner!("wkv7-pretrain-output-forward");

    let tunables = TUNER.init(move || {
        let row_tile_group = TuneGroup::<Wkv7PretrainOutputAutotuneKey>::new("row_tile", |_| 1);
        let mut set = TunableSet::new(key, input_gen);

        for row_tile in PRETRAIN_OUTPUT_ROW_TILE_CANDIDATES {
            set = set.with(
                Tunable::new(
                    &format!("row_tile_{row_tile}"),
                    move |(
                        receptance,
                        weight_decay,
                        replacement_key,
                        value,
                        removal_key_normalized,
                        replacement,
                    )| {
                        Ok::<_, String>(wkv7_pretrain_output_with_row_tile::<R, F>(
                            receptance,
                            weight_decay,
                            replacement_key,
                            value,
                            removal_key_normalized,
                            replacement,
                            chunk_len,
                            row_tile,
                        ))
                    },
                )
                .group(&row_tile_group, move |key| {
                    if is_valid_pretrain_output_row_tile(key, row_tile) {
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
        &CubeTuneId::new(&client, &receptance.device),
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

fn is_valid_pretrain_output_row_tile(key: &Wkv7PretrainOutputAutotuneKey, row_tile: usize) -> bool {
    row_tile <= key.head_size && (row_tile as u32) <= key.hardware.max_units_per_cube
}

fn wkv7_pretrain_output_with_row_tile<R, F>(
    receptance: CubeTensor<R>,
    weight_decay: CubeTensor<R>,
    replacement_key: CubeTensor<R>,
    value: CubeTensor<R>,
    removal_key_normalized: CubeTensor<R>,
    replacement: CubeTensor<R>,
    chunk_len: usize,
    row_tile: usize,
) -> CubeTensor<R>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
{
    let shape = receptance.meta.shape().clone();
    let client = receptance.client.clone();
    let device = receptance.device.clone();
    let output = empty_device::<R, F>(client.clone(), device, shape.clone());
    let dtype = receptance.dtype;

    if shape.num_elements() == 0 {
        return output;
    }

    let row_tiles = shape[3].div_ceil(row_tile);
    let cube_dim = CubeDim::new_1d(row_tile as u32);
    let cube_count = CubeCount::Static(shape[2] as u32, shape[0] as u32, row_tiles as u32);
    let address_type = max_address_type(&[
        &receptance,
        &weight_decay,
        &replacement_key,
        &value,
        &removal_key_normalized,
        &replacement,
        &output,
    ]);

    // SAFETY: The public input contract checks shape/dtype/device/chunk length, and primitive
    // dispatch checks contiguity. The launch grid covers every batch/head pair and every state row.
    unsafe {
        wkv7_pretrain_forward_output_kernel::launch_unchecked::<R>(
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
            shape[1],
            shape[2],
            shape[3],
            row_tile,
            chunk_len,
            dtype.into(),
        );
    }

    output
}

fn wkv7_pretrain<R, F>(inputs: Wkv7PretrainLaunchInputs<R>) -> Wkv7PretrainForwardSaved<R>
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
