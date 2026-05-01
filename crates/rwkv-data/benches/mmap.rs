use criterion::{criterion_group, criterion_main};

#[path = "mmap/bin.rs"]
mod bin;
#[path = "mmap/bin_old.rs"]
mod bin_old;
#[path = "mmap/idx.rs"]
mod idx;
#[path = "mmap/map.rs"]
mod map;
#[path = "mmap/mod.rs"]
mod mmap;
#[path = "mmap/sample.rs"]
mod sample;

criterion_group!(
    benches,
    bin::bin,
    idx::idx,
    map::map,
    sample::sample,
    bin_old::bin_old,
);
criterion_main!(benches);
