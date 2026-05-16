use burn::cubecl;
use cubecl::{cube, prelude::*, std::tensor::layout::linear::LinearView};

const WARP_SIZE: usize = 32;

#[cube]
fn warp_reduce_sum_f32(mut value: f32) -> f32 {
    value += plane_shuffle_xor(value, 16);
    value += plane_shuffle_xor(value, 8);
    value += plane_shuffle_xor(value, 4);
    value += plane_shuffle_xor(value, 2);
    value += plane_shuffle_xor(value, 1);
    value
}

#[cube]
fn block_reduce_sum_f32(
    mut value: f32,
    mut warp_results: SharedMemory<f32>,
    #[comptime] num_warps: usize,
) -> f32 {
    let lane = UNIT_POS_PLANE as usize;
    let warp_id = (UNIT_POS as usize) / WARP_SIZE;

    value = warp_reduce_sum_f32(value);
    if lane == WARP_SIZE - 1 {
        warp_results[warp_id] = value;
    }
    sync_cube();

    if warp_id == 0 {
        let mut warp_value = if lane < num_warps {
            warp_results[lane]
        } else {
            f32::new(0.0)
        };
        warp_value = warp_reduce_sum_f32(warp_value);
        if lane == 0 {
            warp_results[0] = warp_value;
        }
    }
    sync_cube();
    warp_results[0]
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn layer_norm_forward_kernel<F: Float>(
    input: &LinearView<F>,
    gamma: &LinearView<F>,
    beta: &LinearView<F>,
    output: &mut LinearView<F, ReadWrite>,
    d_model: usize,
    epsilon: f32,
    #[comptime] block_size: usize,
    #[comptime] num_warps: usize,
) {
    let row = CUBE_POS_X as usize;
    let row_start = row * d_model;
    let shared = SharedMemory::<f32>::new(num_warps);
    let column_index = RuntimeCell::<usize>::new(UNIT_POS as usize);
    let mut local_sum = f32::new(0.0);
    let mut local_squares = f32::new(0.0);

    while column_index.read() < d_model {
        let value = f32::cast_from(input[row_start + column_index.read()]);
        local_sum += value;
        local_squares += value * value;
        column_index.store(column_index.read() + block_size);
    }

    let sum = block_reduce_sum_f32(local_sum, shared, num_warps);
    let squares = block_reduce_sum_f32(local_squares, shared, num_warps);
    let inv_d_model = f32::new(1.0) / f32::cast_from(d_model);
    let mean = sum * inv_d_model;
    let variance = max(squares * inv_d_model - mean * mean, f32::new(0.0));
    let inv_std = f32::new(1.0) / (variance + epsilon).sqrt();
    let column_index = RuntimeCell::<usize>::new(UNIT_POS as usize);

    while column_index.read() < d_model {
        let column = column_index.read();
        let value = f32::cast_from(input[row_start + column]);
        let normalized = (value - mean) * inv_std;
        output[row_start + column] = F::cast_from(normalized) * gamma[column] + beta[column];
        column_index.store(column + block_size);
    }
}
