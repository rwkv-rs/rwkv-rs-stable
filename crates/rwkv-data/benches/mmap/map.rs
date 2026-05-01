use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion};
use rwkv_data::mmap::map::Map;
use tempfile::tempdir;

use crate::mmap::{
    MAP_WRITE_NUM_SAMPLES,
    MapFileFixture,
    NUM_SAMPLES,
    make_samples,
    offset_entries,
};

pub fn map(c: &mut Criterion) {
    let read_fixture = MapFileFixture::new(NUM_SAMPLES);
    let get_fixture = MapFileFixture::new(NUM_SAMPLES);
    let write_entries = offset_entries(&make_samples(MAP_WRITE_NUM_SAMPLES));
    let read_map = Map::read(&get_fixture.map_path);
    let mut group = c.benchmark_group("rwkv-data/mmap/map");

    group.bench_with_input(
        BenchmarkId::new("push_with_str", write_entries.len()),
        &write_entries,
        |bench, entries| {
            bench.iter_batched(
                || {
                    let dir = tempdir().expect("create mmap map push bench directory");
                    let path = dir.path().join("samples.map");
                    (dir, path)
                },
                |(_dir, path)| {
                    let mut map = Map::new(&path);
                    for entry in entries {
                        map.push_with_str(
                            black_box(entry.line_ref.as_str()),
                            black_box(entry.offset),
                            black_box(entry.length),
                        );
                    }
                    black_box(map.get_with_str(entries[entries.len() - 1].line_ref.as_str()));
                },
                BatchSize::LargeInput,
            );
        },
    );

    group.bench_with_input(
        BenchmarkId::new("push_batch_with_str", write_entries.len()),
        &write_entries,
        |bench, entries| {
            bench.iter_batched(
                || {
                    let dir = tempdir().expect("create mmap map batch push bench directory");
                    let path = dir.path().join("samples.map");
                    (dir, path)
                },
                |(_dir, path)| {
                    let map = Map::new(&path);
                    map.push_batch_with_str(entries.iter().map(|entry| {
                        (
                            black_box(entry.line_ref.as_str()),
                            black_box(entry.offset),
                            black_box(entry.length),
                        )
                    }));
                    black_box(map.get_with_str(entries[entries.len() - 1].line_ref.as_str()));
                },
                BatchSize::LargeInput,
            );
        },
    );

    group.bench_function("read", |bench| {
        bench.iter(|| {
            black_box(Map::read(black_box(&read_fixture.map_path)));
        });
    });

    group.bench_function("get_with_str", |bench| {
        let mut entry_index = 0;

        bench.iter(|| {
            entry_index = (entry_index + 1543) % get_fixture.entries.len();
            let entry = &get_fixture.entries[entry_index];
            black_box(read_map.get_with_str(black_box(entry.line_ref.as_str())));
        });
    });

    group.finish();
}
