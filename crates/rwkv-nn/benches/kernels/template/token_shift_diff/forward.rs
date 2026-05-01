#![allow(non_snake_case)]

use std::hint::black_box;

use burn::{
    prelude::{Backend, Int},
    tensor::{Distribution, Tensor, TensorData},
};
use criterion::{BenchmarkId, Criterion};
use rwkv_nn::kernels::template::token_shift_diff::{
    io::TokenShiftDiffForwardInputs,
    token_shift_diff_custom,
    token_shift_diff_reference,
};

use crate::common;

type B = crate::common::BenchBackend;

pub(crate) fn forward(c: &mut Criterion) {
    let device = crate::common::device::<B>();
    let mut group = c.benchmark_group("rwkv-nn/kernels/token_shift_diff/forward");

    for (name, batch_size, context_len, embedded_dim, full_batch_size) in [
        ("single_token", 4, 1, 512, 9),
        ("small_prefill", 4, 16, 512, 9),
        (
            "infer_prefill",
            8,
            crate::common::CONTEXT_LEN,
            crate::common::EMBEDDED_DIM,
            17,
        ),
    ] {
        let inputs = make_inputs::<B>(
            batch_size,
            context_len,
            embedded_dim,
            full_batch_size,
            &device,
        );

        common::print_speedup_summary::<B, _, _>(
            "rwkv-nn/kernels/token_shift_diff/forward",
            batch_size,
            &device,
            || {
                black_box(token_shift_diff_custom(inputs.clone()));
            },
            || {
                black_box(token_shift_diff_reference(inputs.clone()));
            },
        );

        group.bench_with_input(BenchmarkId::new("custom", name), &name, |bench, _| {
            bench.iter(|| {
                black_box(token_shift_diff_custom(inputs.clone()));
                B::sync(&device).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("baseline", name), &name, |bench, _| {
            bench.iter(|| {
                black_box(token_shift_diff_reference(inputs.clone()));
                B::sync(&device).unwrap();
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
