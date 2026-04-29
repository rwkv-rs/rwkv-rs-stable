#![allow(non_snake_case)]

use std::hint::black_box;

use burn::{Tensor, backend::Autodiff, prelude::Backend, tensor::Distribution};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rwkv_nn::kernels::template::addcmul::{
    addcmul_custom,
    addcmul_reference,
    addcmul5_custom,
    addcmul5_reference,
    io::{Addcmul5ForwardInputs, AddcmulForwardInputs},
};

use crate::common::{CONTEXT_LEN, EMBEDDED_DIM};

#[path = "../../../mod.rs"]
mod common;

type B = common::BenchBackend;
type A = Autodiff<B>;

fn addcmul_backward(c: &mut Criterion) {
    let device = common::device::<A>();
    let mut group = c.benchmark_group("rwkv-nn/kernels/addcmul/backward");

    for bsz in common::BATCH_SIZES {
        let inputs = AddcmulForwardInputs {
            base: Tensor::<A, 3>::random(
                [bsz, CONTEXT_LEN, EMBEDDED_DIM],
                Distribution::Normal(0.0, 1.0),
                &device,
            ),
            diff: Tensor::<A, 3>::random(
                [bsz, CONTEXT_LEN, EMBEDDED_DIM],
                Distribution::Normal(0.0, 1.0),
                &device,
            ),
            scale: Tensor::<A, 3>::random(
                [1, 1, EMBEDDED_DIM],
                Distribution::Normal(0.0, 1.0),
                &device,
            ),
        };
        common::print_speedup_summary::<A, _, _>(
            "rwkv-nn/kernels/addcmul/backward",
            bsz,
            &device,
            || {
                let inputs = AddcmulForwardInputs {
                    base: inputs.base.clone().require_grad(),
                    diff: inputs.diff.clone().require_grad(),
                    scale: inputs.scale.clone().require_grad(),
                };
                black_box(addcmul_custom::<A>(inputs).sum().backward());
            },
            || {
                let inputs = AddcmulForwardInputs {
                    base: inputs.base.clone().require_grad(),
                    diff: inputs.diff.clone().require_grad(),
                    scale: inputs.scale.clone().require_grad(),
                };
                black_box(addcmul_reference(inputs).sum().backward());
            },
        );

        group.bench_with_input(BenchmarkId::new("custom", bsz), &bsz, |bench, _| {
            bench.iter(|| {
                let inputs = AddcmulForwardInputs {
                    base: inputs.base.clone().require_grad(),
                    diff: inputs.diff.clone().require_grad(),
                    scale: inputs.scale.clone().require_grad(),
                };
                black_box(addcmul_custom::<A>(inputs).sum().backward());
                A::sync(&device).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("baseline", bsz), &bsz, |bench, _| {
            bench.iter(|| {
                let inputs = AddcmulForwardInputs {
                    base: inputs.base.clone().require_grad(),
                    diff: inputs.diff.clone().require_grad(),
                    scale: inputs.scale.clone().require_grad(),
                };
                black_box(addcmul_reference(inputs).sum().backward());
                A::sync(&device).unwrap();
            });
        });
    }

    group.finish();
}

fn addcmul5_backward(c: &mut Criterion) {
    let device = common::device::<A>();
    let mut group = c.benchmark_group("rwkv-nn/kernels/addcmul5/backward");

    for bsz in common::BATCH_SIZES {
        let inputs = Addcmul5ForwardInputs {
            base: Tensor::<A, 3>::random(
                [bsz, CONTEXT_LEN, EMBEDDED_DIM],
                Distribution::Normal(0.0, 1.0),
                &device,
            ),
            diff: Tensor::<A, 3>::random(
                [bsz, CONTEXT_LEN, EMBEDDED_DIM],
                Distribution::Normal(0.0, 1.0),
                &device,
            ),
            receptance_scale: Tensor::<A, 3>::random(
                [1, 1, EMBEDDED_DIM],
                Distribution::Normal(0.0, 1.0),
                &device,
            ),
            weight_decay_scale: Tensor::<A, 3>::random(
                [1, 1, EMBEDDED_DIM],
                Distribution::Normal(0.0, 1.0),
                &device,
            ),
            key_scale: Tensor::<A, 3>::random(
                [1, 1, EMBEDDED_DIM],
                Distribution::Normal(0.0, 1.0),
                &device,
            ),
            value_scale: Tensor::<A, 3>::random(
                [1, 1, EMBEDDED_DIM],
                Distribution::Normal(0.0, 1.0),
                &device,
            ),
            learning_rate_scale: Tensor::<A, 3>::random(
                [1, 1, EMBEDDED_DIM],
                Distribution::Normal(0.0, 1.0),
                &device,
            ),
        };
        common::print_speedup_summary::<A, _, _>(
            "rwkv-nn/kernels/addcmul5/backward",
            bsz,
            &device,
            || {
                let output = addcmul5_custom::<A>(Addcmul5ForwardInputs::<A> {
                    base: inputs.base.clone().require_grad(),
                    diff: inputs.diff.clone().require_grad(),
                    receptance_scale: inputs.receptance_scale.clone().require_grad(),
                    weight_decay_scale: inputs.weight_decay_scale.clone().require_grad(),
                    key_scale: inputs.key_scale.clone().require_grad(),
                    value_scale: inputs.value_scale.clone().require_grad(),
                    learning_rate_scale: inputs.learning_rate_scale.clone().require_grad(),
                });
                black_box(
                    (output.receptance_input.sum()
                        + output.weight_decay_input.sum()
                        + output.key_input.sum()
                        + output.value_input.sum()
                        + output.learning_rate_input.sum())
                    .backward(),
                );
            },
            || {
                let output = addcmul5_reference(Addcmul5ForwardInputs {
                    base: inputs.base.clone().require_grad(),
                    diff: inputs.diff.clone().require_grad(),
                    receptance_scale: inputs.receptance_scale.clone().require_grad(),
                    weight_decay_scale: inputs.weight_decay_scale.clone().require_grad(),
                    key_scale: inputs.key_scale.clone().require_grad(),
                    value_scale: inputs.value_scale.clone().require_grad(),
                    learning_rate_scale: inputs.learning_rate_scale.clone().require_grad(),
                });
                black_box(
                    (output.receptance_input.sum()
                        + output.weight_decay_input.sum()
                        + output.key_input.sum()
                        + output.value_input.sum()
                        + output.learning_rate_input.sum())
                    .backward(),
                );
            },
        );

        group.bench_with_input(BenchmarkId::new("custom", bsz), &bsz, |bench, _| {
            bench.iter(|| {
                let output = addcmul5_custom::<A>(Addcmul5ForwardInputs::<A> {
                    base: inputs.base.clone().require_grad(),
                    diff: inputs.diff.clone().require_grad(),
                    receptance_scale: inputs.receptance_scale.clone().require_grad(),
                    weight_decay_scale: inputs.weight_decay_scale.clone().require_grad(),
                    key_scale: inputs.key_scale.clone().require_grad(),
                    value_scale: inputs.value_scale.clone().require_grad(),
                    learning_rate_scale: inputs.learning_rate_scale.clone().require_grad(),
                });
                black_box(
                    (output.receptance_input.sum()
                        + output.weight_decay_input.sum()
                        + output.key_input.sum()
                        + output.value_input.sum()
                        + output.learning_rate_input.sum())
                    .backward(),
                );
                A::sync(&device).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("baseline", bsz), &bsz, |bench, _| {
            bench.iter(|| {
                let output = addcmul5_reference(Addcmul5ForwardInputs {
                    base: inputs.base.clone().require_grad(),
                    diff: inputs.diff.clone().require_grad(),
                    receptance_scale: inputs.receptance_scale.clone().require_grad(),
                    weight_decay_scale: inputs.weight_decay_scale.clone().require_grad(),
                    key_scale: inputs.key_scale.clone().require_grad(),
                    value_scale: inputs.value_scale.clone().require_grad(),
                    learning_rate_scale: inputs.learning_rate_scale.clone().require_grad(),
                });
                black_box(
                    (output.receptance_input.sum()
                        + output.weight_decay_input.sum()
                        + output.key_input.sum()
                        + output.value_input.sum()
                        + output.learning_rate_input.sum())
                    .backward(),
                );
                A::sync(&device).unwrap();
            });
        });
    }

    group.finish();
}

criterion_group!(benches, addcmul_backward, addcmul5_backward);
criterion_main!(benches);
