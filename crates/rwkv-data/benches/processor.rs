use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

#[path = "processor/mod.rs"]
mod processor;

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_secs(1));
    targets = processor::processor
}
criterion_main!(benches);
