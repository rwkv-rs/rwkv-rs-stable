#![allow(non_snake_case)]

use std::hint::black_box;

use burn::{
    backend::Autodiff,
    prelude::{Backend, Int},
    tensor::{Distribution, Tensor, TensorData},
};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rwkv_nn::kernels::template::token_shift_diff::{
    io::TokenShiftDiffForwardInputs,
    token_shift_diff_custom,
    token_shift_diff_reference,
};

use crate::common::{CONTEXT_LEN, EMBEDDED_DIM};

#[path = "../../../mod.rs"]
mod common;

type B = common::BenchBackend;
type A = Autodiff<B>;

fn backward(c: &mut Criterion) {
    let device = common::device::<A>();
    let mut group = c.benchmark_group("rwkv-nn/kernels/token_shift_diff/backward");

    for (name, batch_size, context_len, embedded_dim, full_batch_size) in [
        ("single_token", 4, 1, 512, 9),
        ("small_prefill", 4, 16, 512, 9),
        ("infer_prefill", 8, CONTEXT_LEN, EMBEDDED_DIM, 17),
    ] {
        let inputs = make_inputs::<A>(
            batch_size,
            context_len,
            embedded_dim,
            full_batch_size,
            &device,
        );

        common::print_speedup_summary::<A, _, _>(
            "rwkv-nn/kernels/token_shift_diff/backward",
            batch_size,
            &device,
            || {
                let inputs = require_grad(inputs.clone());
                let output = token_shift_diff_custom(inputs);
                black_box(
                    (output.token_shifted_diff.sum() + output.next_token_shift.sum()).backward(),
                );
            },
            || {
                let inputs = require_grad(inputs.clone());
                let output = token_shift_diff_reference(inputs);
                black_box(
                    (output.token_shifted_diff.sum() + output.next_token_shift.sum()).backward(),
                );
            },
        );

        group.bench_with_input(BenchmarkId::new("custom", name), &name, |bench, _| {
            bench.iter(|| {
                let inputs = require_grad(inputs.clone());
                let output = token_shift_diff_custom(inputs);
                black_box(
                    (output.token_shifted_diff.sum() + output.next_token_shift.sum()).backward(),
                );
                A::sync(&device).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("baseline", name), &name, |bench, _| {
            bench.iter(|| {
                let inputs = require_grad(inputs.clone());
                let output = token_shift_diff_reference(inputs);
                black_box(
                    (output.token_shifted_diff.sum() + output.next_token_shift.sum()).backward(),
                );
                A::sync(&device).unwrap();
            });
        });
    }

    group.finish();
}

fn make_inputs<B: Backend>(
    batch_size: usize,
    context_len: usize,
    embedded_dim: usize,
    full_batch_size: usize,
    device: &B::Device,
) -> TokenShiftDiffForwardInputs<B>
where
    B: rwkv_nn::kernels::template::token_shift_diff::TokenShiftDiffBackend,
{
    let ids = (0..batch_size)
        .map(|batch_index| ((batch_index * 2 + 1) % full_batch_size) as i32)
        .collect::<Vec<_>>();

    TokenShiftDiffForwardInputs {
        embedded_context: Tensor::<B, 3>::random(
            [batch_size, context_len, embedded_dim],
            Distribution::Normal(0.0, 1.0),
            device,
        ),
        embedded_token_shift: Tensor::<B, 2>::random(
            [full_batch_size, embedded_dim],
            Distribution::Normal(0.0, 1.0),
            device,
        ),
        batch_ids: Tensor::<B, 1, Int>::from_ints(TensorData::new(ids, [batch_size]), device),
    }
}

fn require_grad(inputs: TokenShiftDiffForwardInputs<A>) -> TokenShiftDiffForwardInputs<A> {
    TokenShiftDiffForwardInputs {
        embedded_context: inputs.embedded_context.require_grad(),
        embedded_token_shift: inputs.embedded_token_shift.require_grad(),
        batch_ids: inputs.batch_ids,
    }
}

criterion_group!(benches, backward);
criterion_main!(benches);
