use burn::cubecl;
use cubecl::{cube, prelude::*, std::tensor::layout::linear::LinearView};

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn residual_add_forward_kernel<F: Float, N: Size>(
    lhs: &LinearView<Vector<F, N>>,
    rhs: &LinearView<Vector<F, N>>,
    output: &mut LinearView<Vector<F, N>, ReadWrite>,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    output[ABSOLUTE_POS] = lhs[ABSOLUTE_POS] + rhs[ABSOLUTE_POS];
}
