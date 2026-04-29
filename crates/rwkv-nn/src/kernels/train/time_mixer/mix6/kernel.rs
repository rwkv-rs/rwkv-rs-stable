use burn::cubecl;
use cubecl::{cube, prelude::*, std::tensor::layout::linear::LinearView};

#[derive(CubeType, CubeLaunch)]
pub struct Mix6ForwardInputs<F: Float, N: Size> {
    pub embedded_context: LinearView<Vector<F, N>>,
    pub receptance_scale: LinearView<Vector<F, N>>,
    pub weight_decay_scale: LinearView<Vector<F, N>>,
    pub key_scale: LinearView<Vector<F, N>>,
    pub value_scale: LinearView<Vector<F, N>>,
    pub learning_rate_scale: LinearView<Vector<F, N>>,
    pub gate_scale: LinearView<Vector<F, N>>,
}

#[derive(CubeType, CubeLaunch)]
pub struct Mix6ForwardOutputs<F: Float, N: Size> {
    pub receptance_input: LinearView<Vector<F, N>, ReadWrite>,
    pub weight_decay_input: LinearView<Vector<F, N>, ReadWrite>,
    pub key_input: LinearView<Vector<F, N>, ReadWrite>,
    pub value_input: LinearView<Vector<F, N>, ReadWrite>,
    pub learning_rate_input: LinearView<Vector<F, N>, ReadWrite>,
    pub gate_input: LinearView<Vector<F, N>, ReadWrite>,
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn mix6_forward_kernel<F: Float, N: Size>(
    inputs: &Mix6ForwardInputs<F, N>,
    outputs: &mut Mix6ForwardOutputs<F, N>,
    context_len: usize,
) {
    if !outputs.receptance_input.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let embedded_vecs = inputs.receptance_scale.shape();
    let time_index = (ABSOLUTE_POS / embedded_vecs) % context_len;
    let scale_index = ABSOLUTE_POS % embedded_vecs;
    let current = inputs.embedded_context[ABSOLUTE_POS];
    let previous = if time_index == 0 {
        Vector::new(F::new(0.0))
    } else {
        inputs.embedded_context[ABSOLUTE_POS - embedded_vecs]
    };
    let token_shifted_diff = previous - current;

    outputs.receptance_input[ABSOLUTE_POS] =
        current + token_shifted_diff * inputs.receptance_scale[scale_index];
    outputs.weight_decay_input[ABSOLUTE_POS] =
        current + token_shifted_diff * inputs.weight_decay_scale[scale_index];
    outputs.key_input[ABSOLUTE_POS] = current + token_shifted_diff * inputs.key_scale[scale_index];
    outputs.value_input[ABSOLUTE_POS] =
        current + token_shifted_diff * inputs.value_scale[scale_index];
    outputs.learning_rate_input[ABSOLUTE_POS] =
        current + token_shifted_diff * inputs.learning_rate_scale[scale_index];
    outputs.gate_input[ABSOLUTE_POS] =
        current + token_shifted_diff * inputs.gate_scale[scale_index];
}

#[derive(CubeType, CubeLaunch)]
pub struct Mix6StackedForwardOutput<F: Float, N: Size> {
    pub output: LinearView<Vector<F, N>, ReadWrite>,
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn mix6_stacked_forward_kernel<F: Float, N: Size>(
    inputs: &Mix6ForwardInputs<F, N>,
    output: &mut Mix6StackedForwardOutput<F, N>,
    context_len: usize,
    branch_stride: usize,
) {
    if ABSOLUTE_POS >= branch_stride {
        terminate!();
    }

    let embedded_vecs = inputs.receptance_scale.shape();
    let time_index = (ABSOLUTE_POS / embedded_vecs) % context_len;
    let scale_index = ABSOLUTE_POS % embedded_vecs;
    let current = inputs.embedded_context[ABSOLUTE_POS];
    let previous = if time_index == 0 {
        Vector::new(F::new(0.0))
    } else {
        inputs.embedded_context[ABSOLUTE_POS - embedded_vecs]
    };
    let token_shifted_diff = previous - current;

    output.output[ABSOLUTE_POS] =
        current + token_shifted_diff * inputs.receptance_scale[scale_index];
    output.output[branch_stride + ABSOLUTE_POS] =
        current + token_shifted_diff * inputs.weight_decay_scale[scale_index];
    output.output[branch_stride * 2 + ABSOLUTE_POS] =
        current + token_shifted_diff * inputs.key_scale[scale_index];
    output.output[branch_stride * 3 + ABSOLUTE_POS] =
        current + token_shifted_diff * inputs.value_scale[scale_index];
    output.output[branch_stride * 4 + ABSOLUTE_POS] =
        current + token_shifted_diff * inputs.learning_rate_scale[scale_index];
    output.output[branch_stride * 5 + ABSOLUTE_POS] =
        current + token_shifted_diff * inputs.gate_scale[scale_index];
}

#[derive(CubeType, CubeLaunch)]
pub struct Mix6BackwardInputs<F: Float, N: Size> {
    pub output_grad: LinearView<Vector<F, N>>,
    pub embedded_context: LinearView<Vector<F, N>>,
    pub receptance_scale: LinearView<Vector<F, N>>,
    pub weight_decay_scale: LinearView<Vector<F, N>>,
    pub key_scale: LinearView<Vector<F, N>>,
    pub value_scale: LinearView<Vector<F, N>>,
    pub learning_rate_scale: LinearView<Vector<F, N>>,
    pub gate_scale: LinearView<Vector<F, N>>,
}

#[derive(CubeType, CubeLaunch)]
pub struct Mix6BackwardOutputs<F: Float, N: Size> {
    pub embedded_context_grad: LinearView<Vector<F, N>, ReadWrite>,
    pub partial_receptance_scale_grad: LinearView<Vector<F, N>, ReadWrite>,
    pub partial_weight_decay_scale_grad: LinearView<Vector<F, N>, ReadWrite>,
    pub partial_key_scale_grad: LinearView<Vector<F, N>, ReadWrite>,
    pub partial_value_scale_grad: LinearView<Vector<F, N>, ReadWrite>,
    pub partial_learning_rate_scale_grad: LinearView<Vector<F, N>, ReadWrite>,
    pub partial_gate_scale_grad: LinearView<Vector<F, N>, ReadWrite>,
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn mix6_backward_partial_kernel<F: Float, N: Size>(
    inputs: &Mix6BackwardInputs<F, N>,
    outputs: &mut Mix6BackwardOutputs<F, N>,
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
        let zero = Vector::new(F::new(0.0));
        outputs.partial_receptance_scale_grad[partial_index] = zero;
        outputs.partial_weight_decay_scale_grad[partial_index] = zero;
        outputs.partial_key_scale_grad[partial_index] = zero;
        outputs.partial_value_scale_grad[partial_index] = zero;
        outputs.partial_learning_rate_scale_grad[partial_index] = zero;
        outputs.partial_gate_scale_grad[partial_index] = zero;
        terminate!();
    }

    let tentative_end = start_bt + bt_tile;
    let end_bt = if tentative_end < bt_len {
        tentative_end
    } else {
        bt_len
    };
    let branch_stride = bt_len * channel_vecs;
    let one = Vector::new(F::new(1.0));
    let zero = Vector::new(F::new(0.0));
    let receptance_scale = inputs.receptance_scale[channel_vec_index];
    let weight_decay_scale = inputs.weight_decay_scale[channel_vec_index];
    let key_scale = inputs.key_scale[channel_vec_index];
    let value_scale = inputs.value_scale[channel_vec_index];
    let learning_rate_scale = inputs.learning_rate_scale[channel_vec_index];
    let gate_scale = inputs.gate_scale[channel_vec_index];

    let mut receptance_scale_acc = zero;
    let mut weight_decay_scale_acc = zero;
    let mut key_scale_acc = zero;
    let mut value_scale_acc = zero;
    let mut learning_rate_scale_acc = zero;
    let mut gate_scale_acc = zero;
    let mut previous_context = zero;
    let mut has_previous_context = false;
    let bt_index = RuntimeCell::<usize>::new(start_bt);

    while bt_index.read() < end_bt {
        let current_bt = bt_index.read();
        let time_index = current_bt % context_len;
        let tensor_index = current_bt * channel_vecs + channel_vec_index;
        let current_context = inputs.embedded_context[tensor_index];
        let previous = if time_index == 0 {
            zero
        } else if has_previous_context {
            previous_context
        } else {
            inputs.embedded_context[tensor_index - channel_vecs]
        };
        let token_shifted_diff = previous - current_context;

        let receptance_grad = inputs.output_grad[tensor_index];
        let weight_decay_grad = inputs.output_grad[branch_stride + tensor_index];
        let key_grad = inputs.output_grad[branch_stride * 2 + tensor_index];
        let value_grad = inputs.output_grad[branch_stride * 3 + tensor_index];
        let learning_rate_grad = inputs.output_grad[branch_stride * 4 + tensor_index];
        let gate_grad = inputs.output_grad[branch_stride * 5 + tensor_index];

        let mut embedded_context_grad = receptance_grad * (one - receptance_scale)
            + weight_decay_grad * (one - weight_decay_scale)
            + key_grad * (one - key_scale)
            + value_grad * (one - value_scale)
            + learning_rate_grad * (one - learning_rate_scale)
            + gate_grad * (one - gate_scale);

        if time_index + 1 < context_len {
            let next_index = tensor_index + channel_vecs;
            embedded_context_grad = embedded_context_grad
                + inputs.output_grad[next_index] * receptance_scale
                + inputs.output_grad[branch_stride + next_index] * weight_decay_scale
                + inputs.output_grad[branch_stride * 2 + next_index] * key_scale
                + inputs.output_grad[branch_stride * 3 + next_index] * value_scale
                + inputs.output_grad[branch_stride * 4 + next_index] * learning_rate_scale
                + inputs.output_grad[branch_stride * 5 + next_index] * gate_scale;
        }
        outputs.embedded_context_grad[tensor_index] = embedded_context_grad;

        receptance_scale_acc = receptance_scale_acc + receptance_grad * token_shifted_diff;
        weight_decay_scale_acc = weight_decay_scale_acc + weight_decay_grad * token_shifted_diff;
        key_scale_acc = key_scale_acc + key_grad * token_shifted_diff;
        value_scale_acc = value_scale_acc + value_grad * token_shifted_diff;
        learning_rate_scale_acc = learning_rate_scale_acc + learning_rate_grad * token_shifted_diff;
        gate_scale_acc = gate_scale_acc + gate_grad * token_shifted_diff;

        previous_context = current_context;
        has_previous_context = time_index + 1 < context_len;
        bt_index.store(current_bt + 1);
    }

    outputs.partial_receptance_scale_grad[partial_index] = receptance_scale_acc;
    outputs.partial_weight_decay_scale_grad[partial_index] = weight_decay_scale_acc;
    outputs.partial_key_scale_grad[partial_index] = key_scale_acc;
    outputs.partial_value_scale_grad[partial_index] = value_scale_acc;
    outputs.partial_learning_rate_scale_grad[partial_index] = learning_rate_scale_acc;
    outputs.partial_gate_scale_grad[partial_index] = gate_scale_acc;
}

#[derive(CubeType, CubeLaunch)]
pub struct Mix6BackwardFinalizeInputs<F: Float, N: Size> {
    pub partial_receptance_scale_grad: LinearView<Vector<F, N>>,
    pub partial_weight_decay_scale_grad: LinearView<Vector<F, N>>,
    pub partial_key_scale_grad: LinearView<Vector<F, N>>,
    pub partial_value_scale_grad: LinearView<Vector<F, N>>,
    pub partial_learning_rate_scale_grad: LinearView<Vector<F, N>>,
    pub partial_gate_scale_grad: LinearView<Vector<F, N>>,
}

#[derive(CubeType, CubeLaunch)]
pub struct Mix6BackwardFinalizeOutputs<F: Float, N: Size> {
    pub receptance_scale_grad: LinearView<Vector<F, N>, ReadWrite>,
    pub weight_decay_scale_grad: LinearView<Vector<F, N>, ReadWrite>,
    pub key_scale_grad: LinearView<Vector<F, N>, ReadWrite>,
    pub value_scale_grad: LinearView<Vector<F, N>, ReadWrite>,
    pub learning_rate_scale_grad: LinearView<Vector<F, N>, ReadWrite>,
    pub gate_scale_grad: LinearView<Vector<F, N>, ReadWrite>,
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn mix6_backward_finalize_kernel<F: Float, N: Size>(
    inputs: &Mix6BackwardFinalizeInputs<F, N>,
    outputs: &mut Mix6BackwardFinalizeOutputs<F, N>,
    channel_vecs: usize,
    num_bt_tiles: usize,
) {
    let channel_vec_index = ABSOLUTE_POS as usize;
    if channel_vec_index >= channel_vecs {
        terminate!();
    }

    let tile_index = RuntimeCell::<usize>::new(0);
    let mut receptance_scale_acc = Vector::new(F::new(0.0));
    let mut weight_decay_scale_acc = Vector::new(F::new(0.0));
    let mut key_scale_acc = Vector::new(F::new(0.0));
    let mut value_scale_acc = Vector::new(F::new(0.0));
    let mut learning_rate_scale_acc = Vector::new(F::new(0.0));
    let mut gate_scale_acc = Vector::new(F::new(0.0));

    while tile_index.read() < num_bt_tiles {
        let partial_index = tile_index.read() * channel_vecs + channel_vec_index;
        receptance_scale_acc =
            receptance_scale_acc + inputs.partial_receptance_scale_grad[partial_index];
        weight_decay_scale_acc =
            weight_decay_scale_acc + inputs.partial_weight_decay_scale_grad[partial_index];
        key_scale_acc = key_scale_acc + inputs.partial_key_scale_grad[partial_index];
        value_scale_acc = value_scale_acc + inputs.partial_value_scale_grad[partial_index];
        learning_rate_scale_acc =
            learning_rate_scale_acc + inputs.partial_learning_rate_scale_grad[partial_index];
        gate_scale_acc = gate_scale_acc + inputs.partial_gate_scale_grad[partial_index];
        tile_index.store(tile_index.read() + 1);
    }

    outputs.receptance_scale_grad[channel_vec_index] = receptance_scale_acc;
    outputs.weight_decay_scale_grad[channel_vec_index] = weight_decay_scale_acc;
    outputs.key_scale_grad[channel_vec_index] = key_scale_acc;
    outputs.value_scale_grad[channel_vec_index] = value_scale_acc;
    outputs.learning_rate_scale_grad[channel_vec_index] = learning_rate_scale_acc;
    outputs.gate_scale_grad[channel_vec_index] = gate_scale_acc;
}
