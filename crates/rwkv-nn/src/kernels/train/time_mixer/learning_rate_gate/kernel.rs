use burn::cubecl;
use cubecl::{cube, prelude::*, std::tensor::layout::linear::LinearView};

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn learning_rate_gate_forward_kernel<F: Float, N: Size>(
    learning_rate_base: &LinearView<Vector<F, N>>,
    learning_rate_input: &LinearView<Vector<F, N>>,
    learning_rate_output: &mut LinearView<Vector<F, N>, ReadWrite>,
) {
    if !learning_rate_output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let embedded_index = ABSOLUTE_POS % learning_rate_base.shape();
    let pre_activation = learning_rate_base[embedded_index] + learning_rate_input[ABSOLUTE_POS];
    let one = Vector::new(F::new(1.0));
    let zero = Vector::new(F::new(0.0));

    // One vector lane group owns one contiguous embedded-dimension slice. The base vector is
    // broadcast by wrapped embedded-dimension indexing and the sigmoid is computed in the same
    // launch as the add.
    learning_rate_output[ABSOLUTE_POS] = one / (one + (zero - pre_activation).exp());
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn learning_rate_gate_forward_pow2_kernel<F: Float, N: Size>(
    learning_rate_base: &LinearView<Vector<F, N>>,
    learning_rate_input: &LinearView<Vector<F, N>>,
    learning_rate_output: &mut LinearView<Vector<F, N>, ReadWrite>,
    embedded_vec_mask: usize,
) {
    if !learning_rate_output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let embedded_index = ABSOLUTE_POS & embedded_vec_mask;
    let pre_activation = learning_rate_base[embedded_index] + learning_rate_input[ABSOLUTE_POS];
    let one = Vector::new(F::new(1.0));
    let zero = Vector::new(F::new(0.0));

    learning_rate_output[ABSOLUTE_POS] = one / (one + (zero - pre_activation).exp());
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn learning_rate_gate_backward_partial_kernel<F: Float, N: Size>(
    learning_rate_base: &LinearView<Vector<F, N>>,
    learning_rate_input: &LinearView<Vector<F, N>>,
    output_grad: &LinearView<Vector<F, N>>,
    learning_rate_input_grad: &mut LinearView<Vector<F, N>, ReadWrite>,
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
    let learning_rate_base_value = learning_rate_base[channel_vec_index];
    let one = Vector::new(F::new(1.0));
    let zero = Vector::new(F::new(0.0));
    let mut acc = Vector::new(F::new(0.0));

    while bt_index.read() < end_bt {
        let tensor_index = bt_index.read() * channel_vecs + channel_vec_index;
        let pre_activation = learning_rate_base_value + learning_rate_input[tensor_index];
        let learning_rate = one / (one + (zero - pre_activation).exp());
        let pre_activation_grad = output_grad[tensor_index] * learning_rate * (one - learning_rate);

        learning_rate_input_grad[tensor_index] = pre_activation_grad;
        acc = acc + pre_activation_grad;
        bt_index.store(bt_index.read() + 1);
    }

    partial_sums[partial_index] = acc;
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn learning_rate_gate_backward_finalize_kernel<F: Float, N: Size>(
    partial_sums: &LinearView<Vector<F, N>>,
    learning_rate_base_grad: &mut LinearView<Vector<F, N>, ReadWrite>,
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

    learning_rate_base_grad[channel_vec_index] = acc;
}
