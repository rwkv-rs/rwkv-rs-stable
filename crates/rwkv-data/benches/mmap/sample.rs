use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};
use rwkv_data::mmap::sample::{Sampler, calculate_magic_prime};

pub fn sample(c: &mut Criterion) {
    let mut group = c.benchmark_group("rwkv-data/mmap/sample");

    for num_slots in [16_384_u64, 65_536, 262_144] {
        group.bench_with_input(
            BenchmarkId::new("calculate_magic_prime", num_slots),
            &num_slots,
            |bench, &num_slots| {
                bench.iter(|| {
                    black_box(calculate_magic_prime(black_box(num_slots)));
                });
            },
        );
    }

    group.bench_function("sampler_get_base_offset", |bench| {
        let sampler = Sampler::new(8, 3, 16_384, 16_381);
        let mut index = 0;
        let mut mini_epoch_index = 0;

        bench.iter(|| {
            let mut total = 0;

            for _ in 0..1024 {
                total += sampler.get_base_offset(black_box(index), black_box(mini_epoch_index));
                index = (index + 1) % 16_384;
                mini_epoch_index = (mini_epoch_index + 1) % 16;
            }

            black_box(total);
        });
    });

    group.finish();
}
