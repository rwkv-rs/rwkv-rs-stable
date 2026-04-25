use burn::cubecl;
use cubecl::{cube, prelude::*, std::tensor::layout::linear::LinearView};

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn addcmul_forward_kernel<F: Float, N: Size>(
    base: &LinearView<Vector<F, N>>,
    diff: &LinearView<Vector<F, N>>,
    scale: &LinearView<Vector<F, N>>,
    output: &mut LinearView<Vector<F, N>, ReadWrite>,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let channel_index = ABSOLUTE_POS % scale.shape();

    // One vector lane group computes the complete mathematical formula for one contiguous slice of
    // `embedded_dim`. No cross-element communication is needed.
    output[ABSOLUTE_POS] = base[ABSOLUTE_POS] + diff[ABSOLUTE_POS] * scale[channel_index];
}

#[derive(CubeType, CubeLaunch)]
pub struct Addcmul5ForwardInputs<F: Float, N: Size> {
    pub base: LinearView<Vector<F, N>>,
    pub diff: LinearView<Vector<F, N>>,
    pub receptance_scale: LinearView<Vector<F, N>>,
    pub weight_decay_scale: LinearView<Vector<F, N>>,
    pub key_scale: LinearView<Vector<F, N>>,
    pub value_scale: LinearView<Vector<F, N>>,
    pub learning_rate_scale: LinearView<Vector<F, N>>,
}

#[derive(CubeType, CubeLaunch)]
pub struct Addcmul5ForwardOutputs<F: Float, N: Size> {
    pub receptance_output: LinearView<Vector<F, N>, ReadWrite>,
    pub weight_decay_output: LinearView<Vector<F, N>, ReadWrite>,
    pub key_output: LinearView<Vector<F, N>, ReadWrite>,
    pub value_output: LinearView<Vector<F, N>, ReadWrite>,
    pub learning_rate_output: LinearView<Vector<F, N>, ReadWrite>,
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn addcmul5_forward_kernel<F: Float, N: Size>(
    inputs: &Addcmul5ForwardInputs<F, N>,
    outputs: &mut Addcmul5ForwardOutputs<F, N>,
) {
    if !outputs.receptance_output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let channel_index = ABSOLUTE_POS % inputs.receptance_scale.shape();
    let current = inputs.base[ABSOLUTE_POS];
    let delta = inputs.diff[ABSOLUTE_POS];

    // Keep the shared base and diff vectors live once, then combine them with each branch scale.
    // This is the core redundancy removed relative to five independent reference expressions.
    outputs.receptance_output[ABSOLUTE_POS] =
        current + delta * inputs.receptance_scale[channel_index];
    outputs.weight_decay_output[ABSOLUTE_POS] =
        current + delta * inputs.weight_decay_scale[channel_index];
    outputs.key_output[ABSOLUTE_POS] = current + delta * inputs.key_scale[channel_index];
    outputs.value_output[ABSOLUTE_POS] = current + delta * inputs.value_scale[channel_index];
    outputs.learning_rate_output[ABSOLUTE_POS] =
        current + delta * inputs.learning_rate_scale[channel_index];
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn addcmul_backward_diff_kernel<F: Float, N: Size>(
    output_grad: &LinearView<Vector<F, N>>,
    scale: &LinearView<Vector<F, N>>,
    diff_grad: &mut LinearView<Vector<F, N>, ReadWrite>,
) {
    if !diff_grad.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let channel_index = ABSOLUTE_POS % scale.shape();
    diff_grad[ABSOLUTE_POS] = output_grad[ABSOLUTE_POS] * scale[channel_index];
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn addcmul_scale_reduce_partial_kernel<F: Float, N: Size>(
    diff: &LinearView<Vector<F, N>>,
    output_grad: &LinearView<Vector<F, N>>,
    partial_sums: &mut LinearView<Vector<F, N>, ReadWrite>,
    channel_vecs: usize,
    bt_len: usize,
    bt_tile: usize,
) {
    let channel_vec_index = ABSOLUTE_POS_X as usize;
    if channel_vec_index >= channel_vecs {
        terminate!();
    }

    let bt_tile_index = CUBE_POS_Y as usize;
    let start_bt = bt_tile_index * bt_tile;
    let partial_index = bt_tile_index * channel_vecs + channel_vec_index;

    if start_bt >= bt_len {
        partial_sums[partial_index] = Vector::new(F::new(0.0));
        terminate!();
    }

    let tentative_end = start_bt + bt_tile;
    let end_bt = if tentative_end < bt_len {
        tentative_end
    } else {
        bt_len
    };
    let bt_index = RuntimeCell::<usize>::new(start_bt);
    let mut acc = Vector::new(F::new(0.0));

    while bt_index.read() < end_bt {
        let tensor_index = bt_index.read() * channel_vecs + channel_vec_index;
        acc = acc + output_grad[tensor_index] * diff[tensor_index];
        bt_index.store(bt_index.read() + 1);
    }

    partial_sums[partial_index] = acc;
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn addcmul_scale_reduce_finalize_kernel<F: Float, N: Size>(
    partial_sums: &LinearView<Vector<F, N>>,
    scale_grad: &mut LinearView<Vector<F, N>, ReadWrite>,
    channel_vecs: usize,
    num_bt_tiles: usize,
) {
    let channel_vec_index = ABSOLUTE_POS as usize;
    if channel_vec_index >= channel_vecs {
        terminate!();
    }

    let tile_index = RuntimeCell::<usize>::new(0);
    let mut acc = Vector::new(F::new(0.0));

    while tile_index.read() < num_bt_tiles {
        let partial_index = tile_index.read() * channel_vecs + channel_vec_index;
        acc = acc + partial_sums[partial_index];
        tile_index.store(tile_index.read() + 1);
    }

    scale_grad[channel_vec_index] = acc;
}
