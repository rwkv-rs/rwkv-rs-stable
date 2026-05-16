use burn::cubecl;
use cubecl::{cube, prelude::*, std::tensor::layout::linear::LinearView};

#[derive(CubeType, CubeLaunch)]
pub struct KeyPrepareForwardInputs<F: Float> {
    pub key: LinearView<F>,
    pub key_removal: LinearView<F>,
    pub learning_rate: LinearView<F>,
    pub key_replacement: LinearView<F>,
}

#[derive(CubeType, CubeLaunch)]
pub struct KeyPrepareForwardOutputs<F: Float> {
    pub replacement_key: LinearView<F, ReadWrite>,
    pub removal_key_normalized: LinearView<F, ReadWrite>,
    pub replacement: LinearView<F, ReadWrite>,
}

#[derive(CubeType, CubeLaunch)]
pub struct KeyPrepareStackedForwardOutput<F: Float> {
    pub output: LinearView<F, ReadWrite>,
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn key_prepare_forward_kernel<F: Float>(
    inputs: &KeyPrepareForwardInputs<F>,
    outputs: &mut KeyPrepareForwardOutputs<F>,
    embedded_dim: usize,
    head_size: usize,
    head_mask: usize,
) {
    if !outputs.replacement_key.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let embedded_index = ABSOLUTE_POS % embedded_dim;
    let head_lane = if head_mask > 0 {
        embedded_index & head_mask
    } else {
        embedded_index % head_size
    };
    let head_input_start = ABSOLUTE_POS - head_lane;
    let head_scale_start = embedded_index - head_lane;
    let head_lane_cell = RuntimeCell::<usize>::new(0);
    let mut sum_sq = f32::new(0.0);

    while head_lane_cell.read() < head_size {
        let lane = head_lane_cell.read();
        let key_value = f32::cast_from(inputs.key[head_input_start + lane]);
        let scale_value = f32::cast_from(inputs.key_removal[head_scale_start + lane]);
        let removal_key = key_value * scale_value;

        sum_sq += removal_key * removal_key;
        head_lane_cell.store(lane + 1);
    }

    let denominator = sum_sq.sqrt();
    let epsilon = f32::new(1.0e-12);
    let inv_norm = if denominator > epsilon {
        f32::new(1.0) / denominator
    } else {
        f32::new(1.0e12)
    };
    let key_value = f32::cast_from(inputs.key[ABSOLUTE_POS]);
    let learning_rate = f32::cast_from(inputs.learning_rate[ABSOLUTE_POS]);
    let key_removal = f32::cast_from(inputs.key_removal[embedded_index]);
    let key_replacement = f32::cast_from(inputs.key_replacement[embedded_index]);
    let normalized_removal_key = key_value * key_removal * inv_norm;
    let replacement_scale = f32::new(1.0) + (learning_rate - f32::new(1.0)) * key_replacement;

    outputs.replacement_key[ABSOLUTE_POS] = F::cast_from(key_value * replacement_scale);
    outputs.removal_key_normalized[ABSOLUTE_POS] = F::cast_from(-normalized_removal_key);
    outputs.replacement[ABSOLUTE_POS] = F::cast_from(normalized_removal_key * learning_rate);
}

#[cube]
fn warp_sum(mut value: f32) -> f32 {
    value += plane_shuffle_xor(value, 16);
    value += plane_shuffle_xor(value, 8);
    value += plane_shuffle_xor(value, 4);
    value += plane_shuffle_xor(value, 2);
    value += plane_shuffle_xor(value, 1);
    value
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn key_prepare_forward_64_kernel<F: Float>(
    inputs: &KeyPrepareForwardInputs<F>,
    outputs: &mut KeyPrepareForwardOutputs<F>,
    num_heads: usize,
    bth_len: usize,
    #[comptime] warps_per_cube: usize,
) {
    let warp_index = (UNIT_POS as usize) / 32;
    let lane = UNIT_POS_PLANE as usize;
    let bth_index = (CUBE_POS_X as usize) * warps_per_cube + warp_index;
    if bth_index >= bth_len {
        terminate!();
    }

    let head_size = 64;
    let head_index = bth_index % num_heads;
    let base = bth_index * head_size;
    let scale_base = head_index * head_size;
    let lane_offset = lane * 2;
    let index0 = base + lane_offset;
    let index1 = index0 + 1;
    let scale_index0 = scale_base + lane_offset;
    let scale_index1 = scale_index0 + 1;

    let key0 = f32::cast_from(inputs.key[index0]);
    let key1 = f32::cast_from(inputs.key[index1]);
    let key_removal0 = f32::cast_from(inputs.key_removal[scale_index0]);
    let key_removal1 = f32::cast_from(inputs.key_removal[scale_index1]);
    let removal0 = key0 * key_removal0;
    let removal1 = key1 * key_removal1;

    let sum_sq = warp_sum(removal0 * removal0 + removal1 * removal1);
    let denominator = sum_sq.sqrt();
    let epsilon = f32::new(1.0e-12);
    let inv_norm = if denominator > epsilon {
        f32::new(1.0) / denominator
    } else {
        f32::new(1.0e12)
    };

    let learning_rate0 = f32::cast_from(inputs.learning_rate[index0]);
    let learning_rate1 = f32::cast_from(inputs.learning_rate[index1]);
    let key_replacement0 = f32::cast_from(inputs.key_replacement[scale_index0]);
    let key_replacement1 = f32::cast_from(inputs.key_replacement[scale_index1]);
    let normalized0 = removal0 * inv_norm;
    let normalized1 = removal1 * inv_norm;
    let replacement_scale0 = f32::new(1.0) + (learning_rate0 - f32::new(1.0)) * key_replacement0;
    let replacement_scale1 = f32::new(1.0) + (learning_rate1 - f32::new(1.0)) * key_replacement1;

    outputs.replacement_key[index0] = F::cast_from(key0 * replacement_scale0);
    outputs.replacement_key[index1] = F::cast_from(key1 * replacement_scale1);
    outputs.removal_key_normalized[index0] = F::cast_from(-normalized0);
    outputs.removal_key_normalized[index1] = F::cast_from(-normalized1);
    outputs.replacement[index0] = F::cast_from(normalized0 * learning_rate0);
    outputs.replacement[index1] = F::cast_from(normalized1 * learning_rate1);
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn key_prepare_stacked_forward_kernel<F: Float>(
    inputs: &KeyPrepareForwardInputs<F>,
    output: &mut KeyPrepareStackedForwardOutput<F>,
    embedded_dim: usize,
    head_size: usize,
    head_mask: usize,
    branch_stride: usize,
) {
    if ABSOLUTE_POS >= branch_stride {
        terminate!();
    }

    let embedded_index = ABSOLUTE_POS % embedded_dim;
    let head_lane = if head_mask > 0 {
        embedded_index & head_mask
    } else {
        embedded_index % head_size
    };
    let head_input_start = ABSOLUTE_POS - head_lane;
    let head_scale_start = embedded_index - head_lane;
    let head_lane_cell = RuntimeCell::<usize>::new(0);
    let mut sum_sq = f32::new(0.0);

    while head_lane_cell.read() < head_size {
        let lane = head_lane_cell.read();
        let key_value = f32::cast_from(inputs.key[head_input_start + lane]);
        let scale_value = f32::cast_from(inputs.key_removal[head_scale_start + lane]);
        let removal_key = key_value * scale_value;

        sum_sq += removal_key * removal_key;
        head_lane_cell.store(lane + 1);
    }

    let denominator = sum_sq.sqrt();
    let epsilon = f32::new(1.0e-12);
    let inv_norm = if denominator > epsilon {
        f32::new(1.0) / denominator
    } else {
        f32::new(1.0e12)
    };
    let key_value = f32::cast_from(inputs.key[ABSOLUTE_POS]);
    let learning_rate = f32::cast_from(inputs.learning_rate[ABSOLUTE_POS]);
    let key_removal = f32::cast_from(inputs.key_removal[embedded_index]);
    let key_replacement = f32::cast_from(inputs.key_replacement[embedded_index]);
    let normalized_removal_key = key_value * key_removal * inv_norm;
    let replacement_scale = f32::new(1.0) + (learning_rate - f32::new(1.0)) * key_replacement;

    output.output[ABSOLUTE_POS] = F::cast_from(key_value * replacement_scale);
    output.output[branch_stride + ABSOLUTE_POS] = F::cast_from(-normalized_removal_key);
    output.output[branch_stride * 2 + ABSOLUTE_POS] =
        F::cast_from(normalized_removal_key * learning_rate);
}

#[derive(CubeType, CubeLaunch)]
pub struct KeyPrepareBackwardInputs<F: Float> {
    pub output_grad: LinearView<F>,
    pub key: LinearView<F>,
    pub key_removal: LinearView<F>,
    pub learning_rate: LinearView<F>,
    pub key_replacement: LinearView<F>,
}

#[derive(CubeType, CubeLaunch)]
pub struct KeyPrepareBackwardOutputs<F: Float> {
    pub key_grad: LinearView<F, ReadWrite>,
    pub key_removal_grad: LinearView<Atomic<F>, ReadWrite>,
    pub learning_rate_grad: LinearView<F, ReadWrite>,
    pub key_replacement_grad: LinearView<Atomic<F>, ReadWrite>,
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn key_prepare_backward_partial_kernel<F: Float>(
    inputs: &KeyPrepareBackwardInputs<F>,
    outputs: &mut KeyPrepareBackwardOutputs<F>,
    embedded_dim: usize,
    head_size: usize,
    head_mask: usize,
    num_elements: usize,
) {
    if ABSOLUTE_POS >= num_elements {
        terminate!();
    }

    let embedded_index = ABSOLUTE_POS % embedded_dim;
    let head_lane = if head_mask > 0 {
        embedded_index & head_mask
    } else {
        embedded_index % head_size
    };
    let head_input_start = ABSOLUTE_POS - head_lane;
    let head_scale_start = embedded_index - head_lane;
    let head_lane_cell = RuntimeCell::<usize>::new(0);
    let mut sum_sq = f32::new(0.0);

    while head_lane_cell.read() < head_size {
        let lane = head_lane_cell.read();
        let key_value = f32::cast_from(inputs.key[head_input_start + lane]);
        let scale_value = f32::cast_from(inputs.key_removal[head_scale_start + lane]);
        let removal_key = key_value * scale_value;

        sum_sq += removal_key * removal_key;
        head_lane_cell.store(lane + 1);
    }

    let denominator = sum_sq.sqrt();
    let epsilon = f32::new(1.0e-12);
    let inv_norm = if denominator > epsilon {
        f32::new(1.0) / denominator
    } else {
        f32::new(1.0e12)
    };
    let head_lane_cell = RuntimeCell::<usize>::new(0);
    let mut dot = f32::new(0.0);

    while head_lane_cell.read() < head_size {
        let lane = head_lane_cell.read();
        let input_index = head_input_start + lane;
        let scale_index = head_scale_start + lane;
        let key_value = f32::cast_from(inputs.key[input_index]);
        let key_removal = f32::cast_from(inputs.key_removal[scale_index]);
        let learning_rate = f32::cast_from(inputs.learning_rate[input_index]);
        let normalized_removal_key = key_value * key_removal * inv_norm;
        let replacement_grad = f32::cast_from(inputs.output_grad[num_elements * 2 + input_index]);
        let removal_grad = f32::cast_from(inputs.output_grad[num_elements + input_index]);
        let normalized_grad = replacement_grad * learning_rate - removal_grad;

        dot += normalized_grad * normalized_removal_key;
        head_lane_cell.store(lane + 1);
    }

    let key_value = f32::cast_from(inputs.key[ABSOLUTE_POS]);
    let key_removal = f32::cast_from(inputs.key_removal[embedded_index]);
    let learning_rate = f32::cast_from(inputs.learning_rate[ABSOLUTE_POS]);
    let key_replacement = f32::cast_from(inputs.key_replacement[embedded_index]);
    let replacement_key_grad = f32::cast_from(inputs.output_grad[ABSOLUTE_POS]);
    let removal_grad = f32::cast_from(inputs.output_grad[num_elements + ABSOLUTE_POS]);
    let replacement_grad = f32::cast_from(inputs.output_grad[num_elements * 2 + ABSOLUTE_POS]);
    let normalized_removal_key = key_value * key_removal * inv_norm;
    let normalized_grad = replacement_grad * learning_rate - removal_grad;
    let normalized_input_grad = if inv_norm < f32::new(1.0e12) {
        (normalized_grad - normalized_removal_key * dot) * inv_norm
    } else {
        normalized_grad * inv_norm
    };
    let replacement_scale = f32::new(1.0) + (learning_rate - f32::new(1.0)) * key_replacement;

    outputs.key_grad[ABSOLUTE_POS] = F::cast_from(
        replacement_key_grad * replacement_scale + normalized_input_grad * key_removal,
    );
    outputs.key_removal_grad[embedded_index]
        .fetch_add(F::cast_from(normalized_input_grad * key_value));
    outputs.learning_rate_grad[ABSOLUTE_POS] = F::cast_from(
        replacement_key_grad * key_value * key_replacement
            + replacement_grad * normalized_removal_key,
    );
    outputs.key_replacement_grad[embedded_index].fetch_add(F::cast_from(
        replacement_key_grad * key_value * (learning_rate - f32::new(1.0)),
    ));
}

#[derive(CubeType, CubeLaunch)]
pub struct KeyPrepareBackwardFinalizeInputs<F: Float> {
    pub partial_key_removal_grad: LinearView<F>,
    pub partial_key_replacement_grad: LinearView<F>,
}

#[derive(CubeType, CubeLaunch)]
pub struct KeyPrepareBackwardFinalizeOutputs<F: Float> {
    pub key_removal_grad: LinearView<F, ReadWrite>,
    pub key_replacement_grad: LinearView<F, ReadWrite>,
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn key_prepare_backward_finalize_kernel<F: Float>(
    inputs: &KeyPrepareBackwardFinalizeInputs<F>,
    outputs: &mut KeyPrepareBackwardFinalizeOutputs<F>,
    embedded_dim: usize,
    bt_len: usize,
) {
    let embedded_index = ABSOLUTE_POS as usize;
    if embedded_index >= embedded_dim {
        terminate!();
    }

    let bt_index = RuntimeCell::<usize>::new(0);
    let mut key_removal_acc = f32::new(0.0);
    let mut key_replacement_acc = f32::new(0.0);

    while bt_index.read() < bt_len {
        let index = bt_index.read() * embedded_dim + embedded_index;
        key_removal_acc += f32::cast_from(inputs.partial_key_removal_grad[index]);
        key_replacement_acc += f32::cast_from(inputs.partial_key_replacement_grad[index]);
        bt_index.store(bt_index.read() + 1);
    }

    outputs.key_removal_grad[embedded_index] = F::cast_from(key_removal_acc);
    outputs.key_replacement_grad[embedded_index] = F::cast_from(key_replacement_acc);
}
