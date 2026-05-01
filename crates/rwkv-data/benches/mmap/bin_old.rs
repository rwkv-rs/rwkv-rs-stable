use std::hint::black_box;

use criterion::{BenchmarkId, Criterion};
use rwkv_data::mmap::bin_old::BinReader;

use crate::mmap::{NUM_TOKENS, TokenBinFixture};

pub fn bin_old(c: &mut Criterion) {
    let fixture = TokenBinFixture::new(NUM_TOKENS);
    let mut group = c.benchmark_group("rwkv-data/mmap/bin_old");

    group.bench_function("reader_new", |bench| {
        bench.iter(|| {
            black_box(BinReader::<u16>::new(black_box(fixture.legacy_bin_path())));
        });
    });

    for length in [128_u64, 1024, 8192] {
        group.bench_with_input(
            BenchmarkId::new("reader_get_contiguous", length),
            &length,
            |bench, &length| {
                bench.iter(|| {
                    black_box(
                        fixture
                            .legacy_reader
                            .get(black_box(4096), black_box(length)),
                    );
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("reader_get_wraparound", length),
            &length,
            |bench, &length| {
                let offset = fixture.legacy_reader.num_tokens - length / 2;

                bench.iter(|| {
                    black_box(
                        fixture
                            .legacy_reader
                            .get(black_box(offset), black_box(length)),
                    );
                });
            },
        );
    }

    group.finish();
}
