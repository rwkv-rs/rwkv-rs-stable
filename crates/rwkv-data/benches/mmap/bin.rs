use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion};
use rwkv_data::mmap::bin::{BinReader, BinWriter};
use tempfile::tempdir;

use crate::mmap::{NUM_TOKENS, TokenBinFixture, WRITE_NUM_TOKENS, make_tokens};

pub fn bin(c: &mut Criterion) {
    let fixture = TokenBinFixture::new(NUM_TOKENS);
    let write_tokens = make_tokens(WRITE_NUM_TOKENS);
    let mut group = c.benchmark_group("rwkv-data/mmap/bin");

    group.bench_function("writer_push_update_metadata", |bench| {
        bench.iter_batched(
            || {
                let dir = tempdir().expect("create mmap bin writer bench directory");
                let path = dir.path().join("tokens.bin");
                (dir, path)
            },
            |(_dir, path)| {
                let mut writer = BinWriter::<u16>::new(&path, 1, 64 * 1024);
                writer.push(black_box(&write_tokens));
                writer.update_metadata();
                black_box(writer.num_tokens);
            },
            BatchSize::LargeInput,
        );
    });

    group.bench_function("reader_new", |bench| {
        bench.iter(|| {
            black_box(BinReader::<u16>::new(black_box(fixture.bin_path())));
        });
    });

    for length in [128_u64, 1024, 8192] {
        group.bench_with_input(
            BenchmarkId::new("reader_get_contiguous", length),
            &length,
            |bench, &length| {
                bench.iter(|| {
                    black_box(fixture.reader.get(black_box(4096), black_box(length)));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("reader_get_wraparound", length),
            &length,
            |bench, &length| {
                let offset = fixture.reader.num_tokens - length / 2;

                bench.iter(|| {
                    black_box(fixture.reader.get(black_box(offset), black_box(length)));
                });
            },
        );
    }

    group.finish();
}
