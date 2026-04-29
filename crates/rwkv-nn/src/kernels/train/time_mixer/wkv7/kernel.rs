use burn::cubecl;
use cubecl::{cube, prelude::*, std::tensor::layout::linear::LinearView};

#[derive(CubeType, CubeLaunch)]
pub struct Wkv7ForwardInputs<F: Float> {
    pub receptance: LinearView<F>,
    pub weight_decay: LinearView<F>,
    pub replacement_key: LinearView<F>,
    pub value: LinearView<F>,
    pub removal_key_normalized: LinearView<F>,
    pub replacement: LinearView<F>,
}

#[cube]
fn sequence_index(
    batch_index: usize,
    time_index: usize,
    head_index: usize,
    lane_index: usize,
    context_len: usize,
    num_heads: usize,
    head_size: usize,
) -> usize {
    (((batch_index * context_len + time_index) * num_heads + head_index) * head_size) + lane_index
}

#[cube]
fn state_index(
    batch_index: usize,
    head_index: usize,
    row_index: usize,
    col_index: usize,
    num_heads: usize,
    head_size: usize,
) -> usize {
    (((batch_index * num_heads + head_index) * head_size + row_index) * head_size) + col_index
}

#[cube]
fn snapshot_index(
    batch_index: usize,
    head_index: usize,
    chunk_index: usize,
    row_index: usize,
    col_index: usize,
    num_heads: usize,
    num_chunks: usize,
    head_size: usize,
) -> usize {
    ((((batch_index * num_heads + head_index) * num_chunks + chunk_index) * head_size + row_index)
        * head_size)
        + col_index
}

#[cube]
fn decay_from_weight_decay<F: Float>(weight_decay: F) -> F {
    (F::new(0.0) - weight_decay.exp()).exp()
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn wkv7_pretrain_forward_kernel<F: Float>(
    inputs: &Wkv7ForwardInputs<F>,
    output: &mut LinearView<F, ReadWrite>,
    snapshots: &mut LinearView<F, ReadWrite>,
    state_replacement_output: &mut LinearView<F, ReadWrite>,
    context_len: usize,
    num_heads: usize,
    head_size: usize,
    chunk_len: usize,
    #[define(F)] _dtype: StorageType,
) {
    let batch_index = CUBE_POS_Y as usize;
    let head_index = CUBE_POS_X as usize;
    let row_index = UNIT_POS as usize;
    if row_index >= head_size {
        terminate!();
    }

    let num_chunks = context_len / chunk_len;
    let time_cell = RuntimeCell::<usize>::new(0);

    while time_cell.read() < context_len {
        let time_index = time_cell.read();
        let chunk_index = time_index / chunk_len;
        let col_cell = RuntimeCell::<usize>::new(0);
        let mut state_replacement = f32::new(0.0);

        while col_cell.read() < head_size {
            let col_index = col_cell.read();
            let state_value = if time_index == 0 {
                f32::new(0.0)
            } else {
                let previous_chunk_index = if time_index % chunk_len == 0 {
                    chunk_index - 1
                } else {
                    chunk_index
                };
                f32::cast_from(
                    snapshots[snapshot_index(
                        batch_index,
                        head_index,
                        previous_chunk_index,
                        row_index,
                        col_index,
                        num_heads,
                        num_chunks,
                        head_size,
                    )],
                )
            };
            let input_index = sequence_index(
                batch_index,
                time_index,
                head_index,
                col_index,
                context_len,
                num_heads,
                head_size,
            );
            let removal_value: f32 = f32::cast_from(inputs.removal_key_normalized[input_index]);
            state_replacement += state_value * removal_value;
            col_cell.store(col_cell.read() + 1);
        }

        let value_index = sequence_index(
            batch_index,
            time_index,
            head_index,
            row_index,
            context_len,
            num_heads,
            head_size,
        );
        state_replacement_output[value_index] = F::cast_from(state_replacement);
        let value = f32::cast_from(inputs.value[value_index]);
        let col_cell = RuntimeCell::<usize>::new(0);
        let mut output_value = f32::new(0.0);

        while col_cell.read() < head_size {
            let col_index = col_cell.read();
            let input_index = sequence_index(
                batch_index,
                time_index,
                head_index,
                col_index,
                context_len,
                num_heads,
                head_size,
            );
            let state_value = if time_index == 0 {
                f32::new(0.0)
            } else {
                let previous_chunk_index = if time_index % chunk_len == 0 {
                    chunk_index - 1
                } else {
                    chunk_index
                };
                f32::cast_from(
                    snapshots[snapshot_index(
                        batch_index,
                        head_index,
                        previous_chunk_index,
                        row_index,
                        col_index,
                        num_heads,
                        num_chunks,
                        head_size,
                    )],
                )
            };
            let decay: f32 = f32::cast_from(decay_from_weight_decay::<F>(
                inputs.weight_decay[input_index],
            ));
            let replacement: f32 = f32::cast_from(inputs.replacement[input_index]);
            let replacement_key: f32 = f32::cast_from(inputs.replacement_key[input_index]);
            let receptance: f32 = f32::cast_from(inputs.receptance[input_index]);
            let updated =
                state_value * decay + state_replacement * replacement + value * replacement_key;
            snapshots[snapshot_index(
                batch_index,
                head_index,
                chunk_index,
                row_index,
                col_index,
                num_heads,
                num_chunks,
                head_size,
            )] = F::cast_from(updated);
            output_value += updated * receptance;
            col_cell.store(col_cell.read() + 1);
        }

        output[value_index] = F::cast_from(output_value);
        time_cell.store(time_cell.read() + 1);
    }
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn wkv7_state_forward_kernel<F: Float>(
    initial_state: &LinearView<F>,
    inputs: &Wkv7ForwardInputs<F>,
    output: &mut LinearView<F, ReadWrite>,
    next_state: &mut LinearView<F, ReadWrite>,
    snapshots: &mut LinearView<F, ReadWrite>,
    state_replacement_output: &mut LinearView<F, ReadWrite>,
    context_len: usize,
    num_heads: usize,
    head_size: usize,
    chunk_len: usize,
    #[define(F)] _dtype: StorageType,
) {
    let batch_index = CUBE_POS_Y as usize;
    let head_index = CUBE_POS_X as usize;
    let row_index = UNIT_POS as usize;
    if row_index >= head_size {
        terminate!();
    }

    let num_chunks = context_len / chunk_len;
    let time_cell = RuntimeCell::<usize>::new(0);

    while time_cell.read() < context_len {
        let time_index = time_cell.read();
        let chunk_index = time_index / chunk_len;
        let col_cell = RuntimeCell::<usize>::new(0);
        let mut state_replacement = f32::new(0.0);

        while col_cell.read() < head_size {
            let col_index = col_cell.read();
            let state_value = if time_index == 0 {
                f32::cast_from(
                    initial_state[state_index(
                        batch_index,
                        head_index,
                        row_index,
                        col_index,
                        num_heads,
                        head_size,
                    )],
                )
            } else {
                let previous_chunk_index = if time_index % chunk_len == 0 {
                    chunk_index - 1
                } else {
                    chunk_index
                };
                f32::cast_from(
                    snapshots[snapshot_index(
                        batch_index,
                        head_index,
                        previous_chunk_index,
                        row_index,
                        col_index,
                        num_heads,
                        num_chunks,
                        head_size,
                    )],
                )
            };
            let input_index = sequence_index(
                batch_index,
                time_index,
                head_index,
                col_index,
                context_len,
                num_heads,
                head_size,
            );
            let removal_value: f32 = f32::cast_from(inputs.removal_key_normalized[input_index]);
            state_replacement += state_value * removal_value;
            col_cell.store(col_cell.read() + 1);
        }

        let value_index = sequence_index(
            batch_index,
            time_index,
            head_index,
            row_index,
            context_len,
            num_heads,
            head_size,
        );
        state_replacement_output[value_index] = F::cast_from(state_replacement);
        let value = f32::cast_from(inputs.value[value_index]);
        let col_cell = RuntimeCell::<usize>::new(0);
        let mut output_value = f32::new(0.0);

        while col_cell.read() < head_size {
            let col_index = col_cell.read();
            let input_index = sequence_index(
                batch_index,
                time_index,
                head_index,
                col_index,
                context_len,
                num_heads,
                head_size,
            );
            let state_value = if time_index == 0 {
                f32::cast_from(
                    initial_state[state_index(
                        batch_index,
                        head_index,
                        row_index,
                        col_index,
                        num_heads,
                        head_size,
                    )],
                )
            } else {
                let previous_chunk_index = if time_index % chunk_len == 0 {
                    chunk_index - 1
                } else {
                    chunk_index
                };
                f32::cast_from(
                    snapshots[snapshot_index(
                        batch_index,
                        head_index,
                        previous_chunk_index,
                        row_index,
                        col_index,
                        num_heads,
                        num_chunks,
                        head_size,
                    )],
                )
            };
            let decay: f32 = f32::cast_from(decay_from_weight_decay::<F>(
                inputs.weight_decay[input_index],
            ));
            let replacement: f32 = f32::cast_from(inputs.replacement[input_index]);
            let replacement_key: f32 = f32::cast_from(inputs.replacement_key[input_index]);
            let receptance: f32 = f32::cast_from(inputs.receptance[input_index]);
            let updated =
                state_value * decay + state_replacement * replacement + value * replacement_key;
            snapshots[snapshot_index(
                batch_index,
                head_index,
                chunk_index,
                row_index,
                col_index,
                num_heads,
                num_chunks,
                head_size,
            )] = F::cast_from(updated);
            output_value += updated * receptance;
            if time_index + 1 == context_len {
                next_state[state_index(
                    batch_index,
                    head_index,
                    row_index,
                    col_index,
                    num_heads,
                    head_size,
                )] = F::cast_from(updated);
            }
            col_cell.store(col_cell.read() + 1);
        }

        output[value_index] = F::cast_from(output_value);
        time_cell.store(time_cell.read() + 1);
    }
}

#[derive(CubeType, CubeLaunch)]
pub struct Wkv7BackwardInputs<F: Float> {
    pub receptance: LinearView<F>,
    pub weight_decay: LinearView<F>,
    pub replacement_key: LinearView<F>,
    pub value: LinearView<F>,
    pub removal_key_normalized: LinearView<F>,
    pub replacement: LinearView<F>,
    pub output_grad: LinearView<F>,
    pub next_state_grad: LinearView<F>,
    pub snapshots: LinearView<F>,
    pub state_replacement: LinearView<F>,
    pub state_transposed_scratch: LinearView<F, ReadWrite>,
    pub state_grad_scratch: LinearView<F, ReadWrite>,
    pub state_grad_transposed_scratch: LinearView<F, ReadWrite>,
    pub state_replacement_grad_scratch: LinearView<F, ReadWrite>,
}

#[derive(CubeType, CubeLaunch)]
pub struct Wkv7BackwardOutputs<F: Float> {
    pub receptance_grad: LinearView<F, ReadWrite>,
    pub weight_decay_grad: LinearView<F, ReadWrite>,
    pub replacement_key_grad: LinearView<F, ReadWrite>,
    pub value_grad: LinearView<F, ReadWrite>,
    pub removal_key_normalized_grad: LinearView<F, ReadWrite>,
    pub replacement_grad: LinearView<F, ReadWrite>,
    pub initial_state_grad: LinearView<F, ReadWrite>,
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn wkv7_backward_kernel<F: Float>(
    inputs: &mut Wkv7BackwardInputs<F>,
    outputs: &mut Wkv7BackwardOutputs<F>,
    context_len: usize,
    num_heads: usize,
    #[comptime] head_size: usize,
    chunk_len: usize,
    #[define(F)] _dtype: StorageType,
) {
    let batch_index = CUBE_POS_Y as usize;
    let head_index = CUBE_POS_X as usize;
    let lane_index = UNIT_POS as usize;
    let num_chunks = context_len / chunk_len;

    let col_cell = RuntimeCell::<usize>::new(0);
    while col_cell.read() < head_size {
        let col_index = col_cell.read();
        let row_index = state_index(
            batch_index,
            head_index,
            lane_index,
            col_index,
            num_heads,
            head_size,
        );
        inputs.state_transposed_scratch[row_index] = F::cast_from(f32::new(0.0));
        inputs.state_grad_scratch[row_index] = inputs.next_state_grad[state_index(
            batch_index,
            head_index,
            lane_index,
            col_index,
            num_heads,
            head_size,
        )];
        inputs.state_grad_transposed_scratch[row_index] = inputs.next_state_grad[state_index(
            batch_index,
            head_index,
            col_index,
            lane_index,
            num_heads,
            head_size,
        )];
        col_cell.store(col_cell.read() + 1);
    }

    let reverse_cell = RuntimeCell::<usize>::new(context_len);
    while reverse_cell.read() > 0 {
        reverse_cell.store(reverse_cell.read() - 1);
        let time_index = reverse_cell.read();
        let chunk_index = time_index / chunk_len;
        let current_index = sequence_index(
            batch_index,
            time_index,
            head_index,
            lane_index,
            context_len,
            num_heads,
            head_size,
        );

        if (time_index + 1) % chunk_len == 0 {
            let col_cell = RuntimeCell::<usize>::new(0);
            while col_cell.read() < head_size {
                let col_index = col_cell.read();
                inputs.state_transposed_scratch[state_index(
                    batch_index,
                    head_index,
                    lane_index,
                    col_index,
                    num_heads,
                    head_size,
                )] = inputs.snapshots[snapshot_index(
                    batch_index,
                    head_index,
                    chunk_index,
                    col_index,
                    lane_index,
                    num_heads,
                    num_chunks,
                    head_size,
                )];
                col_cell.store(col_cell.read() + 1);
            }
        }

        let mut receptance_grad = f32::new(0.0);
        let col_cell = RuntimeCell::<usize>::new(0);
        while col_cell.read() < head_size {
            let col_index = col_cell.read();
            let col_input_index = sequence_index(
                batch_index,
                time_index,
                head_index,
                col_index,
                context_len,
                num_heads,
                head_size,
            );
            receptance_grad += f32::cast_from(
                inputs.state_transposed_scratch[state_index(
                    batch_index,
                    head_index,
                    lane_index,
                    col_index,
                    num_heads,
                    head_size,
                )],
            ) * f32::cast_from(inputs.output_grad[col_input_index]);
            col_cell.store(col_cell.read() + 1);
        }
        outputs.receptance_grad[current_index] = F::cast_from(receptance_grad);

        let decay_lane = f32::cast_from(decay_from_weight_decay::<F>(
            inputs.weight_decay[current_index],
        ));
        let replacement_key_lane = f32::cast_from(inputs.replacement_key[current_index]);
        let replacement_lane = f32::cast_from(inputs.replacement[current_index]);
        let receptance_lane = f32::cast_from(inputs.receptance[current_index]);
        let output_grad_lane = f32::cast_from(inputs.output_grad[current_index]);
        let inverse_decay = f32::new(1.0) / decay_lane;

        let col_cell = RuntimeCell::<usize>::new(0);
        while col_cell.read() < head_size {
            let col_index = col_cell.read();
            let col_input_index = sequence_index(
                batch_index,
                time_index,
                head_index,
                col_index,
                context_len,
                num_heads,
                head_size,
            );
            let scratch_index = state_index(
                batch_index,
                head_index,
                lane_index,
                col_index,
                num_heads,
                head_size,
            );
            let previous_state = (f32::cast_from(inputs.state_transposed_scratch[scratch_index])
                - replacement_key_lane * f32::cast_from(inputs.value[col_input_index])
                - replacement_lane * f32::cast_from(inputs.state_replacement[col_input_index]))
                * inverse_decay;
            inputs.state_transposed_scratch[scratch_index] = F::cast_from(previous_state);
            inputs.state_grad_scratch[scratch_index] = F::cast_from(
                f32::cast_from(inputs.state_grad_scratch[scratch_index])
                    + output_grad_lane * f32::cast_from(inputs.receptance[col_input_index]),
            );
            inputs.state_grad_transposed_scratch[scratch_index] = F::cast_from(
                f32::cast_from(inputs.state_grad_transposed_scratch[scratch_index])
                    + receptance_lane * f32::cast_from(inputs.output_grad[col_input_index]),
            );
            col_cell.store(col_cell.read() + 1);
        }

        let mut decay_grad = f32::new(0.0);
        let mut replacement_key_grad = f32::new(0.0);
        let mut value_grad = f32::new(0.0);
        let mut state_replacement_grad = f32::new(0.0);
        let mut replacement_grad = f32::new(0.0);
        let col_cell = RuntimeCell::<usize>::new(0);
        while col_cell.read() < head_size {
            let col_index = col_cell.read();
            let col_input_index = sequence_index(
                batch_index,
                time_index,
                head_index,
                col_index,
                context_len,
                num_heads,
                head_size,
            );
            let scratch_index = state_index(
                batch_index,
                head_index,
                lane_index,
                col_index,
                num_heads,
                head_size,
            );
            let state_value = f32::cast_from(inputs.state_transposed_scratch[scratch_index]);
            let state_grad = f32::cast_from(inputs.state_grad_scratch[scratch_index]);
            let state_grad_transposed =
                f32::cast_from(inputs.state_grad_transposed_scratch[scratch_index]);
            decay_grad += state_grad_transposed * state_value;
            replacement_key_grad +=
                state_grad_transposed * f32::cast_from(inputs.value[col_input_index]);
            value_grad += state_grad * f32::cast_from(inputs.replacement_key[col_input_index]);
            state_replacement_grad +=
                state_grad * f32::cast_from(inputs.replacement[col_input_index]);
            replacement_grad +=
                state_grad_transposed * f32::cast_from(inputs.state_replacement[col_input_index]);
            col_cell.store(col_cell.read() + 1);
        }

        let weight_decay = f32::cast_from(inputs.weight_decay[current_index]);
        outputs.weight_decay_grad[current_index] =
            F::cast_from(decay_grad * (f32::new(0.0) - weight_decay.exp() * decay_lane));
        outputs.replacement_key_grad[current_index] = F::cast_from(replacement_key_grad);
        outputs.value_grad[current_index] = F::cast_from(value_grad);
        outputs.replacement_grad[current_index] = F::cast_from(replacement_grad);

        inputs.state_replacement_grad_scratch
            [state_index(batch_index, head_index, lane_index, 0, num_heads, head_size)] =
            F::cast_from(state_replacement_grad);

        let mut removal_key_normalized_grad = f32::new(0.0);
        let col_cell = RuntimeCell::<usize>::new(0);
        while col_cell.read() < head_size {
            let col_index = col_cell.read();
            let state_replacement_grad_index =
                state_index(batch_index, head_index, col_index, 0, num_heads, head_size);
            removal_key_normalized_grad += f32::cast_from(
                inputs.state_transposed_scratch[state_index(
                    batch_index,
                    head_index,
                    lane_index,
                    col_index,
                    num_heads,
                    head_size,
                )],
            ) * f32::cast_from(
                inputs.state_replacement_grad_scratch[state_replacement_grad_index],
            );
            col_cell.store(col_cell.read() + 1);
        }
        outputs.removal_key_normalized_grad[current_index] =
            F::cast_from(removal_key_normalized_grad);

        let removal_key_normalized_lane =
            f32::cast_from(inputs.removal_key_normalized[current_index]);
        let col_cell = RuntimeCell::<usize>::new(0);
        while col_cell.read() < head_size {
            let col_index = col_cell.read();
            let col_input_index = sequence_index(
                batch_index,
                time_index,
                head_index,
                col_index,
                context_len,
                num_heads,
                head_size,
            );
            let state_replacement_grad_index =
                state_index(batch_index, head_index, col_index, 0, num_heads, head_size);
            let scratch_index = state_index(
                batch_index,
                head_index,
                lane_index,
                col_index,
                num_heads,
                head_size,
            );
            inputs.state_grad_scratch[scratch_index] = F::cast_from(
                f32::cast_from(inputs.state_grad_scratch[scratch_index])
                    * f32::cast_from(decay_from_weight_decay::<F>(
                        inputs.weight_decay[col_input_index],
                    ))
                    + state_replacement_grad
                        * f32::cast_from(inputs.removal_key_normalized[col_input_index]),
            );
            inputs.state_grad_transposed_scratch[scratch_index] = F::cast_from(
                f32::cast_from(inputs.state_grad_transposed_scratch[scratch_index]) * decay_lane
                    + removal_key_normalized_lane
                        * f32::cast_from(
                            inputs.state_replacement_grad_scratch[state_replacement_grad_index],
                        ),
            );
            col_cell.store(col_cell.read() + 1);
        }
    }

    let col_cell = RuntimeCell::<usize>::new(0);
    while col_cell.read() < head_size {
        let col_index = col_cell.read();
        outputs.initial_state_grad[state_index(
            batch_index,
            head_index,
            lane_index,
            col_index,
            num_heads,
            head_size,
        )] = inputs.state_grad_scratch[state_index(
            batch_index,
            head_index,
            lane_index,
            col_index,
            num_heads,
            head_size,
        )];
        col_cell.store(col_cell.read() + 1);
    }
}
