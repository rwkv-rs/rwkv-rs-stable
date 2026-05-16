use burn::cubecl;
use cubecl::{cube, prelude::*, std::tensor::layout::linear::LinearView};

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn weight_decay_transform_forward_kernel<F: Float, N: Size>(
    weight_decay_base: &LinearView<Vector<F, N>>,
    weight_decay_input: &LinearView<Vector<F, N>>,
    output: &mut LinearView<Vector<F, N>, ReadWrite>,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let embedded_index = ABSOLUTE_POS % weight_decay_base.shape();
    let pre_activation = weight_decay_base[embedded_index] + weight_decay_input[ABSOLUTE_POS];
    let one = Vector::new(F::new(1.0));
    let half = Vector::new(F::new(0.5));
    let zero = Vector::new(F::new(0.0));

    output[ABSOLUTE_POS] = zero - (one + (zero - pre_activation).exp()).ln() - half;
}

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn weight_decay_transform_forward_pow2_kernel<F: Float, N: Size>(
    weight_decay_base: &LinearView<Vector<F, N>>,
    weight_decay_input: &LinearView<Vector<F, N>>,
    output: &mut LinearView<Vector<F, N>, ReadWrite>,
    embedded_vec_mask: usize,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let embedded_index = ABSOLUTE_POS & embedded_vec_mask;
    let pre_activation = weight_decay_base[embedded_index] + weight_decay_input[ABSOLUTE_POS];
    let one = Vector::new(F::new(1.0));
    let half = Vector::new(F::new(0.5));
    let zero = Vector::new(F::new(0.0));

    output[ABSOLUTE_POS] = zero - (one + (zero - pre_activation).exp()).ln() - half;
}
