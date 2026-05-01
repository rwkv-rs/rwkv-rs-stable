use criterion::{criterion_group, criterion_main};

#[path = "tokenizer/mod.rs"]
mod tokenizer;

criterion_group!(benches, tokenizer::tokenizer);
criterion_main!(benches);
