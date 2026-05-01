use burn::cubecl;
use cubecl::{cube, prelude::*};

#[derive(CubeLaunch, CubeType)]
pub struct TokenShiftDiffInputs<F: Float> {
    pub embedded_context: Tensor<F>,
    pub batch_ids: Tensor<u32>,
}

#[derive(CubeLaunch, CubeType)]
pub struct TokenShiftDiffOutputs<F: Float> {
    pub token_shifted_diff: Tensor<F>,
    pub next_token_shift: Tensor<F>,
}

#[cube(launch)]
pub fn fused_token_shift_diff_kernel<F: Float>(
    inputs: &TokenShiftDiffInputs<F>,
    outputs: &mut TokenShiftDiffOutputs<F>,
) {
    let index = ABSOLUTE_POS;
    let batch_size = inputs.embedded_context.shape(0);
    let context_len = inputs.embedded_context.shape(1);
    let embedded_dim = inputs.embedded_context.shape(2);

    if index >= batch_size * embedded_dim {
        terminate!();
    }

    let feature_index = index % embedded_dim;
    let batch_index = index / embedded_dim;
    let state_index = inputs.batch_ids[batch_index] as usize;
    let shift_index = state_index * embedded_dim + feature_index;

    let mut previous = outputs.next_token_shift[shift_index];

    for time_index in 0..context_len {
        let output_index = (batch_index * context_len + time_index) * embedded_dim + feature_index;
        let current = inputs.embedded_context[output_index];
        outputs.token_shifted_diff[output_index] = previous - current;
        previous = current;
    }

    outputs.next_token_shift[shift_index] = previous;
}
