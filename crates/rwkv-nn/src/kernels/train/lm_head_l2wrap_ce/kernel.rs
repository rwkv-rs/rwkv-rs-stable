use burn::cubecl;
use cubecl::{cube, prelude::*, std::tensor::layout::linear::LinearView};

const WARP_SIZE: usize = 32;

#[derive(CubeType, CubeLaunch)]
pub struct LmHeadL2WrapCeInputs<F: Float, I: Int> {
    pub logits: LinearView<F>,
    pub targets: LinearView<I>,
}

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
fn warp_reduce_max_f32(mut value: f32) -> f32 {
    value = max(value, plane_shuffle_xor(value, 16));
    value = max(value, plane_shuffle_xor(value, 8));
    value = max(value, plane_shuffle_xor(value, 4));
    value = max(value, plane_shuffle_xor(value, 2));
    value = max(value, plane_shuffle_xor(value, 1));
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

#[cube]
fn block_reduce_max_f32(
    mut value: f32,
    mut warp_results: SharedMemory<f32>,
    #[comptime] num_warps: usize,
) -> f32 {
    let lane = UNIT_POS_PLANE as usize;
    let warp_id = (UNIT_POS as usize) / WARP_SIZE;

    value = warp_reduce_max_f32(value);
    if lane == WARP_SIZE - 1 {
        warp_results[warp_id] = value;
    }
    sync_cube();

    if warp_id == 0 {
        let mut warp_value = if lane < num_warps {
            warp_results[lane]
        } else {
            f32::new(-f32::MAX)
        };
        warp_value = warp_reduce_max_f32(warp_value);
        if lane == 0 {
            warp_results[0] = warp_value;
        }
    }
    sync_cube();
    warp_results[0]
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn lm_head_l2wrap_ce_forward_row_kernel<F: Float, I: Int>(
    inputs: &LmHeadL2WrapCeInputs<F, I>,
    row_losses: &mut LinearView<F, ReadWrite>,
    vocab_size: usize,
    num_tokens: usize,
    #[comptime] block_size: usize,
    #[comptime] num_warps: usize,
) {
    let row = CUBE_POS_X as usize;
    if row >= num_tokens {
        terminate!();
    }

    let row_start = row * vocab_size;
    let target = usize::cast_from(inputs.targets[row]);
    let shared = SharedMemory::<f32>::new(num_warps);
    let mut max_logit = f32::new(-f32::MAX);
    let column_index = RuntimeCell::<usize>::new(UNIT_POS as usize);

    while column_index.read() < vocab_size {
        let value = f32::cast_from(inputs.logits[row_start + column_index.read()]);
        if value > max_logit {
            max_logit = value;
        }
        column_index.store(column_index.read() + block_size);
    }

    let row_max = block_reduce_max_f32(max_logit, shared, num_warps);
    let column_index = RuntimeCell::<usize>::new(UNIT_POS as usize);
    let mut local_sum = f32::new(0.0);

    while column_index.read() < vocab_size {
        let column = column_index.read();
        let value = f32::cast_from(inputs.logits[row_start + column]);
        local_sum += (value - row_max).exp();
        column_index.store(column + block_size);
    }

    let denominator = block_reduce_sum_f32(local_sum, shared, num_warps);

    if UNIT_POS == 0 {
        row_losses[row] = if target < vocab_size {
            let target_logit = f32::cast_from(inputs.logits[row_start + target]);
            F::cast_from(denominator.ln() + row_max - target_logit)
        } else {
            F::new(0.0)
        };
    }
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn lm_head_l2wrap_ce_forward_finalize_kernel<F: Float>(
    partial_sums: &LinearView<F>,
    loss: &mut LinearView<F, ReadWrite>,
    num_partials: usize,
    num_tokens: usize,
    #[comptime] num_warps: usize,
) {
    let partial_index = RuntimeCell::<usize>::new(UNIT_POS as usize);
    let mut acc = f32::new(0.0);

    while partial_index.read() < num_partials {
        acc += f32::cast_from(partial_sums[partial_index.read()]);
        partial_index.store(partial_index.read() + CUBE_DIM_X as usize);
    }
    let shared = SharedMemory::<f32>::new(num_warps);
    let acc = block_reduce_sum_f32(acc, shared, num_warps);

    if UNIT_POS == 0 {
        loss[0] = F::cast_from(acc / f32::cast_from(num_tokens));
    }
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn lm_head_l2wrap_ce_backward_kernel<F: Float, I: Int>(
    inputs: &LmHeadL2WrapCeInputs<F, I>,
    output_grad: &LinearView<F>,
    logits_grad: &mut LinearView<F, ReadWrite>,
    vocab_size: usize,
    num_tokens: usize,
    l2wrap_factor: f32,
    #[comptime] block_size: usize,
    #[comptime] num_warps: usize,
) {
    let row = CUBE_POS_X as usize;
    if row >= num_tokens {
        terminate!();
    }

    let row_start = row * vocab_size;
    let shared = SharedMemory::<f32>::new(num_warps);
    let mut max_logit = f32::new(-f32::MAX);
    let column_index = RuntimeCell::<usize>::new(UNIT_POS as usize);

    while column_index.read() < vocab_size {
        let value = f32::cast_from(inputs.logits[row_start + column_index.read()]);
        if value > max_logit {
            max_logit = value;
        }
        column_index.store(column_index.read() + block_size);
    }

    let row_max = block_reduce_max_f32(max_logit, shared, num_warps);
    let column_index = RuntimeCell::<usize>::new(UNIT_POS as usize);
    let mut local_sum = f32::new(0.0);

    while column_index.read() < vocab_size {
        let value = f32::cast_from(inputs.logits[row_start + column_index.read()]);
        local_sum += (value - row_max).exp();
        column_index.store(column_index.read() + block_size);
    }

    let denominator = block_reduce_sum_f32(local_sum, shared, num_warps);
    let row_lse = denominator.ln() + row_max;
    let target = usize::cast_from(inputs.targets[row]);
    let inv_tokens = f32::new(1.0) / f32::cast_from(num_tokens);
    let output_grad_value = f32::cast_from(output_grad[0]);
    let column_index = RuntimeCell::<usize>::new(UNIT_POS as usize);

    while column_index.read() < vocab_size {
        let column = column_index.read();
        let index = row_start + column;
        let value = f32::cast_from(inputs.logits[index]);
        let softmax = (value - row_lse).exp();
        let target_grad = if column == target {
            f32::new(1.0)
        } else {
            f32::new(0.0)
        };
        let ce_grad = (softmax - target_grad) * inv_tokens;
        let l2wrap_grad = if value == row_max {
            row_max * l2wrap_factor * inv_tokens
        } else {
            f32::new(0.0)
        };

        logits_grad[index] = F::cast_from((ce_grad + l2wrap_grad) * output_grad_value);
        column_index.store(column + block_size);
    }
}
