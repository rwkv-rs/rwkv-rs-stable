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
pub fn gated_readout_combine_forward_kernel<F: Float>(
    wkv_output: &LinearView<F>,
    norm_gamma: &LinearView<F>,
    norm_beta: &LinearView<F>,
    gate: &LinearView<F>,
    receptance: &LinearView<F>,
    replacement_key: &LinearView<F>,
    value: &LinearView<F>,
    bonus: &LinearView<F>,
    output: &mut LinearView<F, ReadWrite>,
    head_size: usize,
    actual_head_size: usize,
    embedded_dim: usize,
    epsilon: f32,
    #[comptime] num_warps: usize,
) {
    let row = CUBE_POS_X as usize;
    let head = CUBE_POS_Y as usize;
    let lane = UNIT_POS as usize;
    let head_start = (row * (embedded_dim / head_size) + head) * head_size;
    let output_start = row * embedded_dim + head * head_size;
    let bonus_start = head * head_size;
    let shared = SharedMemory::<f32>::new(num_warps);
    let mut local_sum = f32::new(0.0);
    let mut local_value = f32::new(0.0);

    if lane < actual_head_size {
        let index = head_start + lane;
        local_value = f32::cast_from(wkv_output[index]);
        local_sum = local_value;
    }

    let sum = block_reduce_sum_f32(local_sum, shared, num_warps);
    let inv_head_size = f32::new(1.0) / f32::cast_from(actual_head_size);
    let mean = sum * inv_head_size;
    let mut local_variance = f32::new(0.0);
    let mut local_bonus = f32::new(0.0);

    if lane < actual_head_size {
        let index = head_start + lane;
        let centered = local_value - mean;
        local_variance = centered * centered;
        local_bonus = f32::cast_from(receptance[index])
            * f32::cast_from(replacement_key[index])
            * f32::cast_from(bonus[bonus_start + lane]);
    }

    let variance = block_reduce_sum_f32(local_variance, shared, num_warps) * inv_head_size;
    let inv_std = f32::new(1.0) / (variance + epsilon).sqrt();
    let bonus_sum = block_reduce_sum_f32(local_bonus, shared, num_warps);

    if lane < actual_head_size {
        let index = head_start + lane;
        let output_index = output_start + lane;
        let affine_index = head * head_size + lane;
        let normalized = F::cast_from((local_value - mean) * inv_std) * norm_gamma[affine_index]
            + norm_beta[affine_index];
        output[output_index] =
            (normalized + F::cast_from(bonus_sum) * value[index]) * gate[output_index];
    }
}
