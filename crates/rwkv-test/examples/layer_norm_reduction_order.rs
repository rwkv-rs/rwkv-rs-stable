use std::{env, fs, path::PathBuf};

use anyhow::{Context, Result, bail};
use half::{bf16, f16};
use safetensors::tensor::{Dtype, SafeTensors, TensorView};

const D_MODEL: usize = 768;
const EPSILON: f32 = 1.0e-5;
const WARP_SIZE: usize = 32;

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let fixture = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/rwkv-test/test_data/rwkv_lm/bf16/case_000000"));
    let weights = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("weights/rwkv-init-0.1b-ctx512-test.st"));

    run_case(
        "layer_norm0",
        &fixture,
        "embedding/embedded_context.safetensors",
        Some("layer_norm0/embedded_context.safetensors"),
        &weights,
        &["layer_norm_for_first_cell.gamma", "blocks.0.ln0.weight"],
        &["layer_norm_for_first_cell.beta", "blocks.0.ln0.bias"],
    )?;
    run_case(
        "cell0_pre_time_mix_ln1",
        &fixture,
        "layer_norm0/embedded_context.safetensors",
        None,
        &weights,
        &[
            "cells.cells.0.pre_layer_norm_for_time_mix.gamma",
            "blocks.0.ln1.weight",
        ],
        &[
            "cells.cells.0.pre_layer_norm_for_time_mix.beta",
            "blocks.0.ln1.bias",
        ],
    )?;
    run_case(
        "cell0_pre_channel_mix_ln2",
        &fixture,
        "cells/cell_0000/embedded_context_after_time_mixer.safetensors",
        None,
        &weights,
        &[
            "cells.cells.0.pre_layer_norm_for_channel_mix.gamma",
            "blocks.0.ln2.weight",
        ],
        &[
            "cells.cells.0.pre_layer_norm_for_channel_mix.beta",
            "blocks.0.ln2.bias",
        ],
    )?;

    Ok(())
}

fn run_case(
    label: &str,
    fixture: &PathBuf,
    input_path: &str,
    expected_path: Option<&str>,
    weights: &PathBuf,
    gamma_names: &[&str],
    beta_names: &[&str],
) -> Result<()> {
    let (_, input_shape, input) = rwkv_test::read_safetensor_values(fixture.join(input_path))?;
    let expected = if let Some(path) = expected_path {
        let (_, expected_shape, expected) = rwkv_test::read_safetensor_values(fixture.join(path))?;
        if expected_shape != input_shape {
            bail!("expected output shape {input_shape:?}, got {expected_shape:?}");
        }
        Some(expected.into_iter().map(|v| v as f32).collect::<Vec<_>>())
    } else {
        None
    };
    let gamma = read_first_named_tensor(weights, gamma_names)?;
    let beta = read_first_named_tensor(weights, beta_names)?;

    if input_shape.last().copied() != Some(D_MODEL) {
        bail!("expected input d_model {D_MODEL}, got shape {input_shape:?}");
    }
    if gamma.len() != D_MODEL || beta.len() != D_MODEL {
        bail!(
            "expected gamma/beta length {D_MODEL}, got gamma={} beta={}",
            gamma.len(),
            beta.len()
        );
    }

    let rows = input.len() / D_MODEL;
    let input: Vec<f32> = input.into_iter().map(|v| v as f32).collect();
    let gamma: Vec<f32> = gamma.into_iter().map(|v| v as f32).collect();
    let beta: Vec<f32> = beta.into_iter().map(|v| v as f32).collect();

    let mut sum_256 = Stats::new();
    let mut sum_ordered = Stats::new();
    let mut squares_256 = Stats::new();
    let mut squares_ordered = Stats::new();
    let mut mean_256 = Stats::new();
    let mut mean_ordered = Stats::new();
    let mut inv_std_256 = Stats::new();
    let mut inv_std_ordered = Stats::new();
    let mut norm_256 = Stats::new();
    let mut norm_ordered = Stats::new();
    let mut output_1024_expected = Stats::new();
    let mut output_256_expected = Stats::new();
    let mut output_ordered_expected = Stats::new();
    let mut output_256_1024 = Stats::new();
    let mut output_ordered_1024 = Stats::new();

    for row in 0..rows {
        let row_values = &input[row * D_MODEL..(row + 1) * D_MODEL];
        let safe = moments_block(row_values, 1024);
        let ordinary = moments_block(row_values, 256);
        let ordered = moments_ordered_256(row_values);

        sum_256.push(ordinary.sum, safe.sum);
        sum_ordered.push(ordered.sum, safe.sum);
        squares_256.push(ordinary.squares, safe.squares);
        squares_ordered.push(ordered.squares, safe.squares);
        mean_256.push(ordinary.mean, safe.mean);
        mean_ordered.push(ordered.mean, safe.mean);
        inv_std_256.push(ordinary.inv_std, safe.inv_std);
        inv_std_ordered.push(ordered.inv_std, safe.inv_std);

        for col in 0..D_MODEL {
            let index = row * D_MODEL + col;
            let value = row_values[col];
            let safe_norm = (value - safe.mean) * safe.inv_std;
            let ordinary_norm = (value - ordinary.mean) * ordinary.inv_std;
            let ordered_norm = (value - ordered.mean) * ordered.inv_std;

            norm_256.push(ordinary_norm, safe_norm);
            norm_ordered.push(ordered_norm, safe_norm);

            let safe_output = layer_norm_output(safe_norm, gamma[col], beta[col]);
            let ordinary_output = layer_norm_output(ordinary_norm, gamma[col], beta[col]);
            let ordered_output = layer_norm_output(ordered_norm, gamma[col], beta[col]);

            if let Some(expected) = expected.as_ref() {
                output_1024_expected.push(safe_output, expected[index]);
                output_256_expected.push(ordinary_output, expected[index]);
                output_ordered_expected.push(ordered_output, expected[index]);
            }
            output_256_1024.push(ordinary_output, safe_output);
            output_ordered_1024.push(ordered_output, safe_output);
        }
    }

    println!("case={label}");
    println!("fixture={}", fixture.display());
    println!("input={input_path}");
    println!("weights={}", weights.display());
    println!("rows={rows} d_model={D_MODEL} epsilon={EPSILON}");
    print_stats("sum ordinary256_vs_1024", sum_256);
    print_stats("sum ordered256_vs_1024", sum_ordered);
    print_stats("squares ordinary256_vs_1024", squares_256);
    print_stats("squares ordered256_vs_1024", squares_ordered);
    print_stats("mean ordinary256_vs_1024", mean_256);
    print_stats("mean ordered256_vs_1024", mean_ordered);
    print_stats("inv_std ordinary256_vs_1024", inv_std_256);
    print_stats("inv_std ordered256_vs_1024", inv_std_ordered);
    print_stats("normalized ordinary256_vs_1024", norm_256);
    print_stats("normalized ordered256_vs_1024", norm_ordered);
    if expected.is_some() {
        print_stats("output safe1024_vs_expected", output_1024_expected);
        print_stats("output ordinary256_vs_expected", output_256_expected);
        print_stats("output ordered256_vs_expected", output_ordered_expected);
    }
    print_stats("output ordinary256_vs_1024", output_256_1024);
    print_stats("output ordered256_vs_1024", output_ordered_1024);
    println!();

    Ok(())
}

fn layer_norm_output(normalized: f32, gamma: f32, beta: f32) -> f32 {
    bf16::from_f32(bf16::from_f32(normalized).to_f32() * gamma + beta).to_f32()
}

#[derive(Clone, Copy)]
struct Moments {
    sum: f32,
    squares: f32,
    mean: f32,
    inv_std: f32,
}

fn moments_block(row: &[f32], block_size: usize) -> Moments {
    let mut local_sum = vec![0.0f32; block_size];
    let mut local_squares = vec![0.0f32; block_size];

    for unit in 0..block_size {
        let mut col = unit;
        while col < row.len() {
            let value = row[col];
            local_sum[unit] += value;
            local_squares[unit] += value * value;
            col += block_size;
        }
    }

    finish(block_reduce(&local_sum), block_reduce(&local_squares))
}

fn moments_ordered_256(row: &[f32]) -> Moments {
    let mut partial_sum = [0.0f32; WARP_SIZE];
    let mut partial_squares = [0.0f32; WARP_SIZE];

    for partial in 0..24 {
        let start = partial * WARP_SIZE;
        let mut sum_lanes = [0.0f32; WARP_SIZE];
        let mut square_lanes = [0.0f32; WARP_SIZE];
        for lane in 0..WARP_SIZE {
            let value = row[start + lane];
            sum_lanes[lane] = value;
            square_lanes[lane] = value * value;
        }
        partial_sum[partial] = warp_reduce_lane(warp_reduce(sum_lanes), WARP_SIZE - 1);
        partial_squares[partial] = warp_reduce_lane(warp_reduce(square_lanes), WARP_SIZE - 1);
    }

    finish(
        warp_reduce_lane(warp_reduce(partial_sum), 0),
        warp_reduce_lane(warp_reduce(partial_squares), 0),
    )
}

fn finish(sum: f32, squares: f32) -> Moments {
    let inv_d_model = 1.0 / D_MODEL as f32;
    let mean = sum * inv_d_model;
    let variance = (squares * inv_d_model - mean * mean).max(0.0);
    let inv_std = 1.0 / (variance + EPSILON).sqrt();
    Moments {
        sum,
        squares,
        mean,
        inv_std,
    }
}

fn block_reduce(values: &[f32]) -> f32 {
    let num_warps = values.len() / WARP_SIZE;
    let mut warp_results = [0.0f32; WARP_SIZE];
    for warp in 0..num_warps {
        let mut lanes = [0.0f32; WARP_SIZE];
        lanes.copy_from_slice(&values[warp * WARP_SIZE..(warp + 1) * WARP_SIZE]);
        warp_results[warp] = warp_reduce_lane(warp_reduce(lanes), WARP_SIZE - 1);
    }
    warp_reduce_lane(warp_reduce(warp_results), 0)
}

fn warp_reduce(mut lanes: [f32; WARP_SIZE]) -> [f32; WARP_SIZE] {
    for offset in [16, 8, 4, 2, 1] {
        let previous = lanes;
        for lane in 0..WARP_SIZE {
            lanes[lane] = previous[lane] + previous[lane ^ offset];
        }
    }
    lanes
}

fn warp_reduce_lane(lanes: [f32; WARP_SIZE], lane: usize) -> f32 {
    lanes[lane]
}

#[derive(Clone, Copy)]
struct Stats {
    count: usize,
    max_abs: f32,
    max_index: usize,
    actual_at_max: f32,
    expected_at_max: f32,
    sum_abs: f64,
}

impl Stats {
    fn new() -> Self {
        Self {
            count: 0,
            max_abs: 0.0,
            max_index: 0,
            actual_at_max: 0.0,
            expected_at_max: 0.0,
            sum_abs: 0.0,
        }
    }

    fn push(&mut self, actual: f32, expected: f32) {
        let abs = (actual - expected).abs();
        if abs > self.max_abs {
            self.max_abs = abs;
            self.max_index = self.count;
            self.actual_at_max = actual;
            self.expected_at_max = expected;
        }
        self.sum_abs += f64::from(abs);
        self.count += 1;
    }

    fn mean_abs(&self) -> f64 {
        self.sum_abs / self.count as f64
    }
}

fn print_stats(name: &str, stats: Stats) {
    println!(
        "{name}: count={} max_abs={:.9e} mean_abs={:.9e} max_index={} actual={:.9e} expected={:.9e}",
        stats.count,
        stats.max_abs,
        stats.mean_abs(),
        stats.max_index,
        stats.actual_at_max,
        stats.expected_at_max
    );
}

fn read_first_named_tensor(path: &PathBuf, names: &[&str]) -> Result<Vec<f64>> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let tensors = SafeTensors::deserialize(&bytes)
        .with_context(|| format!("failed to parse {}", path.display()))?;

    for name in names {
        if let Ok(view) = tensors.tensor(name) {
            return decode(view).with_context(|| format!("failed to decode tensor {name}"));
        }
    }

    bail!("missing any of tensors {names:?} in {}", path.display())
}

fn decode(view: TensorView<'_>) -> Result<Vec<f64>> {
    let raw = view.data();
    match view.dtype() {
        Dtype::F64 => Ok(chunks(raw, 8, |b| f64::from_le_bytes(arr(b)))),
        Dtype::F32 => Ok(chunks(raw, 4, |b| f32::from_le_bytes(arr(b)) as f64)),
        Dtype::F16 => Ok(chunks(raw, 2, |b| {
            f16::from_bits(u16::from_le_bytes(arr(b))).to_f64()
        })),
        Dtype::BF16 => Ok(chunks(raw, 2, |b| {
            bf16::from_bits(u16::from_le_bytes(arr(b))).to_f64()
        })),
        dtype => bail!("unsupported tensor dtype {dtype}"),
    }
}

fn chunks(raw: &[u8], size: usize, f: impl Fn(&[u8]) -> f64) -> Vec<f64> {
    raw.chunks_exact(size).map(f).collect()
}

fn arr<const N: usize>(bytes: &[u8]) -> [u8; N] {
    bytes.try_into().unwrap()
}
