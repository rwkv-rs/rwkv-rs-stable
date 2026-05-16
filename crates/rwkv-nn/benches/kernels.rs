#[cfg(feature = "bench-profile")]
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

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

fn criterion_config() -> Criterion {
    let criterion = Criterion::default();

    #[cfg(feature = "bench-profile")]
    let criterion = criterion
        .sample_size(10)
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_millis(500));

    criterion
}

criterion_group!(
    name = benches;
    config = criterion_config();
    targets =
    addcmul_forward::addcmul_forward,
    addcmul_forward::addcmul5_forward,
    addcmul_backward::addcmul_backward,
    addcmul_backward::addcmul5_backward,
    token_shift_diff_forward::forward,
    token_shift_diff_backward::backward,
);
criterion_main!(benches);
