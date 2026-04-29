use burn::cubecl;
use cubecl::{cube, prelude::*, std::tensor::layout::linear::LinearView};

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn channel_mixer_mix_forward_kernel<F: Float, N: Size>(
    embedded_context: &LinearView<Vector<F, N>>,
    key_scale: &LinearView<Vector<F, N>>,
    context_len: usize,
    key_input: &mut LinearView<Vector<F, N>, ReadWrite>,
) {
    if !key_input.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let embedded_dim = key_scale.shape();
    let embedded_index = ABSOLUTE_POS % embedded_dim;
    let time_index = (ABSOLUTE_POS / embedded_dim) % context_len;
    let current = embedded_context[ABSOLUTE_POS];
    let previous = if time_index == 0 {
        Vector::new(F::new(0.0))
    } else {
        embedded_context[ABSOLUTE_POS - embedded_dim]
    };

    // Each lane group owns one contiguous embedded-dimension slice. The scale vector is broadcast
    // by embedded-dimension indexing, and the previous token stays in the same batch row because
    // the first time position explicitly uses zero.
    key_input[ABSOLUTE_POS] = current + (previous - current) * key_scale[embedded_index];
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn channel_mixer_relu_square_forward_kernel<F: Float, N: Size>(
    pre_activation: &LinearView<Vector<F, N>>,
    activated_key: &mut LinearView<Vector<F, N>, ReadWrite>,
) {
    if !activated_key.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let relu = max(pre_activation[ABSOLUTE_POS], Vector::new(F::new(0.0)));

    // The activation is independent per key projection element, so one vector lane group computes
    // the ReLU and square for a contiguous slice of the expanded dimension.
    activated_key[ABSOLUTE_POS] = relu * relu;
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn channel_mixer_relu_square_backward_from_output_kernel<F: Float, N: Size>(
    activated_key_grad: &LinearView<Vector<F, N>>,
    activated_key: &LinearView<Vector<F, N>>,
    key_projection_grad: &mut LinearView<Vector<F, N>, ReadWrite>,
) {
    if !key_projection_grad.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let activation = max(activated_key[ABSOLUTE_POS], Vector::new(F::new(0.0)));

    // CUDA cmix v5 stores the squared ReLU output and derives the backward multiplier from
    // `sqrt(activated_key)`, avoiding a checkpoint for the pre-activation projection.
    key_projection_grad[ABSOLUTE_POS] =
        activated_key_grad[ABSOLUTE_POS] * activation.sqrt() * Vector::new(F::new(2.0));
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn channel_mixer_mix_backward_partial_kernel<F: Float, N: Size>(
    mixed_grad: &LinearView<Vector<F, N>>,
    embedded_context: &LinearView<Vector<F, N>>,
    key_scale: &LinearView<Vector<F, N>>,
    embedded_context_grad: &mut LinearView<Vector<F, N>, ReadWrite>,
    partial_key_scale_grad: &mut LinearView<Vector<F, N>, ReadWrite>,
    channel_vecs: usize,
    bt_len: usize,
    context_len: usize,
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
        partial_key_scale_grad[partial_index] = Vector::new(F::new(0.0));
        terminate!();
    }

    let tentative_end = start_bt + bt_tile;
    let end_bt = if tentative_end < bt_len {
        tentative_end
    } else {
        bt_len
    };
    let scale = key_scale[channel_vec_index];
    let one_minus_scale = Vector::new(F::new(1.0)) - scale;
    let bt_index = RuntimeCell::<usize>::new(start_bt);
    let mut previous_context = Vector::new(F::new(0.0));
    let mut has_previous_context = false;
    let mut acc = Vector::new(F::new(0.0));

    while bt_index.read() < end_bt {
        let current_bt = bt_index.read();
        let time_index = current_bt % context_len;
        let tensor_index = current_bt * channel_vecs + channel_vec_index;
        let grad = mixed_grad[tensor_index];
        let current_context = embedded_context[tensor_index];

        let previous = if time_index == 0 {
            Vector::new(F::new(0.0))
        } else if has_previous_context {
            previous_context
        } else {
            embedded_context[tensor_index - channel_vecs]
        };
        let next_grad = if time_index + 1 < context_len {
            mixed_grad[tensor_index + channel_vecs]
        } else {
            Vector::new(F::new(0.0))
        };

        embedded_context_grad[tensor_index] = grad * one_minus_scale + next_grad * scale;
        acc = acc + grad * (previous - current_context);

        previous_context = current_context;
        has_previous_context = time_index + 1 < context_len;
        bt_index.store(current_bt + 1);
    }

    partial_key_scale_grad[partial_index] = acc;
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn channel_mixer_key_scale_reduce_finalize_kernel<F: Float, N: Size>(
    partial_key_scale_grad: &LinearView<Vector<F, N>>,
    key_scale_grad: &mut LinearView<Vector<F, N>, ReadWrite>,
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
        acc = acc + partial_key_scale_grad[partial_index];
        tile_index.store(tile_index.read() + 1);
    }

    key_scale_grad[channel_vec_index] = acc;
}
