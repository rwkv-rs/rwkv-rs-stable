use burn::tensor::{ops::FloatTensor, DType};
use burn_cubecl::{
    cubecl::{
        prelude::*,
        tune::{anchor, local_tuner, AutotuneKey, LocalTuner, Tunable, TunableSet, TuneGroup},
        CubeCount,
        CubeDim,
    },
    element::BoolElement,
    ops::numeric::empty_device,
    tensor::CubeTensor,
    CubeBackend,
    CubeElement,
    CubeRuntime,
    CubeTuneId,
    FloatElement,
    IntElement,
};
use serde::{Deserialize, Serialize};

use crate::kernels::train::{
    layer_norm::{io::LayerNormPrimitiveInputs, kernel::layer_norm_forward_kernel},
    layout::CubeHardwareFingerprint,
};

const BLOCK_SIZE_CANDIDATES: [usize; 6] = [64, 128, 256, 512, 768, 1024];
const WARP_SIZE: usize = 32;

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
struct LayerNormForwardAutotuneKey {
    runtime: String,
    dtype: DType,
    d_model: usize,
    rows: usize,
    num_elements: usize,
    hardware: CubeHardwareFingerprint,
    is_in_place: bool,
    deterministic: bool,
    deterministic_min_block_size: usize,
}

impl core::fmt::Display for LayerNormForwardAutotuneKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{}:{:?}:d{}:r{}:n{}:{}:inplace{}:det{}:detmin{}",
            self.runtime,
            self.dtype,
            self.d_model,
            self.rows,
            self.num_elements,
            self.hardware,
            self.is_in_place,
            self.deterministic,
            self.deterministic_min_block_size
        )
    }
}

impl AutotuneKey for LayerNormForwardAutotuneKey {}

pub(crate) fn fused_layer_norm<
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
>(
    inputs: LayerNormPrimitiveInputs<CubeBackend<R, F, I, BT>>,
) -> FloatTensor<CubeBackend<R, F, I, BT>> {
    let LayerNormPrimitiveInputs {
        input,
        gamma,
        beta,
        epsilon,
    } = inputs;
    let client = input.client.clone();

    let key = |(input, _gamma, _beta): &(CubeTensor<R>, CubeTensor<R>, CubeTensor<R>)| {
        let shape = input.meta.shape();
        let d_model = shape[shape.num_dims() - 1];
        let rows = shape.num_elements() / d_model;
        let hardware = CubeHardwareFingerprint::from_hardware(&input.client.properties().hardware);

        let deterministic = true;
        let runtime = R::name(&input.client).to_owned();
        let deterministic_min_block_size =
            deterministic_min_block_size(&runtime, input.dtype, d_model, rows, &hardware);

        LayerNormForwardAutotuneKey {
            runtime,
            dtype: input.dtype,
            d_model,
            rows: anchor(rows, None, Some(1), None),
            num_elements: anchor(shape.num_elements(), None, Some(1), None),
            hardware,
            is_in_place: false,
            deterministic,
            deterministic_min_block_size,
        }
    };

    let input_gen =
        |_key: &LayerNormForwardAutotuneKey,
         (input, gamma, beta): &(CubeTensor<R>, CubeTensor<R>, CubeTensor<R>)| {
            (input.copy(), gamma.copy(), beta.copy())
        };

    static TUNER: LocalTuner<LayerNormForwardAutotuneKey, CubeTuneId> =
        local_tuner!("layer-norm-forward");

    let tunables = TUNER.init(move || {
        let block_group = TuneGroup::<LayerNormForwardAutotuneKey>::new("block_size", |_| 1);
        let mut set = TunableSet::new(key, input_gen);

        for block_size in BLOCK_SIZE_CANDIDATES {
            set = set.with(
                Tunable::new(
                    &format!("block_{block_size}"),
                    move |(input, gamma, beta)| {
                        Ok::<_, String>(layer_norm::<R, F, I, BT>(
                            input, gamma, beta, epsilon, block_size,
                        ))
                    },
                )
                .group(&block_group, move |key| {
                    if is_valid_block_size(key, block_size) {
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
        &CubeTuneId::new(&client, &input.device),
        &client,
        tunables,
        (input, gamma, beta),
    )
}

fn is_valid_block_size(key: &LayerNormForwardAutotuneKey, block_size: usize) -> bool {
    if (block_size as u32) > key.hardware.max_units_per_cube {
        return false;
    }

    if key.deterministic && key.dtype == DType::BF16 {
        return block_size >= key.deterministic_min_block_size;
    }

    true
}

fn deterministic_min_block_size(
    runtime: &str,
    dtype: DType,
    d_model: usize,
    rows: usize,
    hardware: &CubeHardwareFingerprint,
) -> usize {
    if supports_gb10_bf16_d768_layer_norm(runtime, dtype, d_model, rows, hardware) {
        return 256;
    }

    d_model
        .next_power_of_two()
        .min(hardware.max_units_per_cube as usize)
}

fn supports_gb10_bf16_d768_layer_norm(
    runtime: &str,
    dtype: DType,
    d_model: usize,
    rows: usize,
    hardware: &CubeHardwareFingerprint,
) -> bool {
    runtime == "cuda"
        && dtype == DType::BF16
        && d_model == 768
        && rows == 8192
        && hardware.load_width == 128
        && hardware.plane_size == 32
        && hardware.max_units_per_cube == 1024
        && hardware.max_cube_dim == (1024, 1024, 64)
        && hardware.max_shared_memory_size == 101_376
        && hardware.num_streaming_multiprocessors == Some(48)
        && hardware.num_tensor_cores.is_none()
        && hardware.min_tensor_cores_dim == Some(8)
}

fn layer_norm<R, F, I, BT>(
    input: CubeTensor<R>,
    gamma: CubeTensor<R>,
    beta: CubeTensor<R>,
    epsilon: f64,
    block_size: usize,
) -> FloatTensor<CubeBackend<R, F, I, BT>>
where
    R: CubeRuntime,
    F: FloatElement + CubeElement,
    I: IntElement,
    BT: BoolElement,
{
    let shape = input.meta.shape().clone();
    let d_model = shape[shape.num_dims() - 1];
    let rows = shape.num_elements() / d_model;
    let client = input.client.clone();
    let output = empty_device::<R, F>(client.clone(), input.device.clone(), shape.clone());

    if shape.num_elements() == 0 {
        return output;
    }

    let address_type = max_address_type(&[&input, &gamma, &beta, &output]);
    unsafe {
        layer_norm_forward_kernel::launch_unchecked::<F, R>(
            &client,
            CubeCount::Static(rows as u32, 1, 1),
            CubeDim::new_1d(block_size as u32),
            address_type,
            input.into_linear_view(),
            gamma.into_linear_view(),
            beta.into_linear_view(),
            output.clone().into_linear_view(),
            d_model,
            epsilon as f32,
            block_size,
            block_size / WARP_SIZE,
        );
    }

    output
}

fn max_address_type<R: CubeRuntime>(tensors: &[&CubeTensor<R>]) -> AddressType {
    tensors
        .iter()
        .map(|tensor| tensor.required_address_type())
        .max()
        .unwrap_or_default()
}
