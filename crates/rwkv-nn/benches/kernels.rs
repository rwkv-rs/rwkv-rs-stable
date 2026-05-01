use criterion::{criterion_group, criterion_main};

#[path = "mod.rs"]
mod common;

#[path = "kernels/template/addcmul/backward.rs"]
mod addcmul_backward;
#[path = "kernels/template/addcmul/forward.rs"]
mod addcmul_forward;
#[path = "kernels/template/token_shift_diff/backward.rs"]
mod token_shift_diff_backward;
#[path = "kernels/template/token_shift_diff/forward.rs"]
mod token_shift_diff_forward;

criterion_group!(
    benches,
    addcmul_forward::addcmul_forward,
    addcmul_forward::addcmul5_forward,
    addcmul_backward::addcmul_backward,
    addcmul_backward::addcmul5_backward,
    token_shift_diff_forward::forward,
    token_shift_diff_backward::backward,
);
criterion_main!(benches);
