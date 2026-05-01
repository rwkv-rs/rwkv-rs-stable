use std::hint::black_box;

use criterion::{BatchSize, Criterion};
use rwkv_data::mmap::idx::{IdxReader, IdxWriter};

use crate::mmap::{BenchSample, IdxWriteFixture, NUM_SAMPLES, SampleFixture, WRITE_NUM_SAMPLES};

pub fn idx(c: &mut Criterion) {
    let fixture = SampleFixture::new(NUM_SAMPLES);
    let mut group = c.benchmark_group("rwkv-data/mmap/idx");

    group.bench_function("writer_push_update_metadata", |bench| {
        bench.iter_batched(
            || IdxWriteFixture::new(WRITE_NUM_SAMPLES),
            |fixture| {
                let mut writer = IdxWriter::new(&fixture.idx_path, 64 * 1024);
                for sample in &fixture.samples {
                    writer.push(black_box(sample), black_box(&fixture.map));
                }
                writer.update_metadata();
                black_box(writer.num_samples);
            },
            BatchSize::LargeInput,
        );
    });

    group.bench_function("reader_new", |bench| {
        bench.iter(|| {
            black_box(IdxReader::<BenchSample>::new(black_box(fixture.idx_path())));
        });
    });

    group.bench_function("reader_get", |bench| {
        let mut sample_index = 0;

        bench.iter(|| {
            sample_index = (sample_index + 1543) % fixture.idx_reader.num_samples;
            black_box(
                fixture
                    .idx_reader
                    .get(black_box(sample_index), black_box(&fixture.bin_reader)),
            );
        });
    });

    group.finish();
}
