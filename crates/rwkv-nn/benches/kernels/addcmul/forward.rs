#![allow(non_snake_case)]

use std::hint::black_box;

use burn::{prelude::Backend, tensor::Distribution, Tensor};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rwkv_nn::kernels::addcmul::{
    addcmul5_custom,
    addcmul5_reference,
    addcmul_custom,
    addcmul_reference,
    io::{Addcmul5ForwardInputs, AddcmulForwardInputs},
};

use crate::common::{CONTEXT_LEN, EMBEDDED_DIM};

#[path = "../../mod.rs"]
mod common;

type B = common::BenchBackend;

fn addcmul_forward(c: &mut Criterion) {
    let device = common::device::<B>();
    let mut group = c.benchmark_group("rwkv-nn/kernels/addcmul/forward");

    for bsz in common::BATCH_SIZES {
        let inputs = AddcmulForwardInputs {
            base: Tensor::<B, 3>::random(
                [bsz, CONTEXT_LEN, EMBEDDED_DIM],
                Distribution::Normal(0.0, 1.0),
                &device,
            ),
            diff: Tensor::<B, 3>::random(
                [bsz, CONTEXT_LEN, EMBEDDED_DIM],
                Distribution::Normal(0.0, 1.0),
                &device,
            ),
            scale: Tensor::<B, 3>::random(
                [1, 1, EMBEDDED_DIM],
                Distribution::Normal(0.0, 1.0),
                &device,
            ),
        };
        common::print_speedup_summary::<B, _, _>(
            "rwkv-nn/kernels/addcmul/forward",
            bsz,
            &device,
            || {
                black_box(addcmul_custom(inputs.clone()));
            },
            || {
                black_box(addcmul_reference(inputs.clone()));
            },
        );

        group.bench_with_input(BenchmarkId::new("custom", bsz), &bsz, |bench, _| {
            bench.iter(|| {
                black_box(addcmul_custom(inputs.clone()));
                B::sync(&device).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("baseline", bsz), &bsz, |bench, _| {
            bench.iter(|| {
                black_box(addcmul_reference(inputs.clone()));
                B::sync(&device).unwrap();
            });
        });
    }

    group.finish();
}

fn addcmul5_forward(c: &mut Criterion) {
    let device = common::device::<B>();
    let mut group = c.benchmark_group("rwkv-nn/kernels/addcmul5/forward");

    for bsz in common::BATCH_SIZES {
        let inputs = Addcmul5ForwardInputs {
            base: Tensor::<B, 3>::random(
                [bsz, CONTEXT_LEN, EMBEDDED_DIM],
                Distribution::Normal(0.0, 1.0),
                &device,
            ),
            diff: Tensor::<B, 3>::random(
                [bsz, CONTEXT_LEN, EMBEDDED_DIM],
                Distribution::Normal(0.0, 1.0),
                &device,
            ),
            receptance_scale: Tensor::<B, 3>::random(
                [1, 1, EMBEDDED_DIM],
                Distribution::Normal(0.0, 1.0),
                &device,
            ),
            weight_decay_scale: Tensor::<B, 3>::random(
                [1, 1, EMBEDDED_DIM],
                Distribution::Normal(0.0, 1.0),
                &device,
            ),
            key_scale: Tensor::<B, 3>::random(
                [1, 1, EMBEDDED_DIM],
                Distribution::Normal(0.0, 1.0),
                &device,
            ),
            value_scale: Tensor::<B, 3>::random(
                [1, 1, EMBEDDED_DIM],
                Distribution::Normal(0.0, 1.0),
                &device,
            ),
            learning_rate_scale: Tensor::<B, 3>::random(
                [1, 1, EMBEDDED_DIM],
                Distribution::Normal(0.0, 1.0),
                &device,
            ),
        };
        common::print_speedup_summary::<B, _, _>(
            "rwkv-nn/kernels/addcmul5/forward",
            bsz,
            &device,
            || {
                black_box(addcmul5_custom(inputs.clone()));
            },
            || {
                black_box(addcmul5_reference(inputs.clone()));
            },
        );

        group.bench_with_input(BenchmarkId::new("custom", bsz), &bsz, |bench, _| {
            bench.iter(|| {
                black_box(addcmul5_custom(inputs.clone()));
                B::sync(&device).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("baseline", bsz), &bsz, |bench, _| {
            bench.iter(|| {
                black_box(addcmul5_reference(inputs.clone()));
                B::sync(&device).unwrap();
            });
        });
    }

    group.finish();
}

criterion_group!(benches, addcmul_forward, addcmul5_forward);
criterion_main!(benches);
