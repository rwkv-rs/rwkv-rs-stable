use std::{
    fs::File,
    io::Write,
    path::{Path, PathBuf},
};

use rwkv_data::mmap::{bin::BinWriter, map::Map};
use rwkv_derive::LineRef;
use tempfile::{TempDir, tempdir};

pub const NUM_TOKENS: usize = 1 << 20;
pub const NUM_SAMPLES: usize = 16_384;
pub const MAP_WRITE_NUM_SAMPLES: usize = 16;
pub const WRITE_NUM_TOKENS: usize = 64 * 1024;
pub const WRITE_NUM_SAMPLES: usize = 1_024;

#[derive(Clone, Debug, LineRef)]
pub struct BenchSample {
    pub id: u64,
    pub score: u64,
    #[line_ref]
    pub text: String,
}

pub struct TokenBinFixture {
    _dir: TempDir,
    bin_path: PathBuf,
    legacy_bin_path: PathBuf,
    pub reader: rwkv_data::mmap::bin::BinReader<u16>,
    pub legacy_reader: rwkv_data::mmap::bin_old::BinReader<u16>,
}

impl TokenBinFixture {
    pub fn new(num_tokens: usize) -> Self {
        let dir = tempdir().expect("create mmap token fixture directory");
        let bin_path = dir.path().join("tokens.bin");
        let legacy_bin_path = dir.path().join("legacy.bin");
        let tokens = make_tokens(num_tokens);

        write_bin_file(&bin_path, &tokens);
        write_legacy_bin_files(&legacy_bin_path, &tokens);

        let reader = rwkv_data::mmap::bin::BinReader::<u16>::new(&bin_path);
        let legacy_reader = rwkv_data::mmap::bin_old::BinReader::<u16>::new(&legacy_bin_path);

        Self {
            _dir: dir,
            bin_path,
            legacy_bin_path,
            reader,
            legacy_reader,
        }
    }

    pub fn bin_path(&self) -> &Path {
        &self.bin_path
    }

    pub fn legacy_bin_path(&self) -> &Path {
        &self.legacy_bin_path
    }
}

pub struct SampleFixture {
    files: SampleFiles,
    pub bin_reader: rwkv_data::mmap::bin::BinReader<u8>,
    pub idx_reader: rwkv_data::mmap::idx::IdxReader<BenchSample>,
}

impl SampleFixture {
    pub fn new(num_samples: usize) -> Self {
        let files = SampleFiles::new(num_samples);
        let bin_reader = rwkv_data::mmap::bin::BinReader::<u8>::new(&files.bin_path);
        let idx_reader = rwkv_data::mmap::idx::IdxReader::<BenchSample>::new(&files.idx_path);

        Self {
            files,
            bin_reader,
            idx_reader,
        }
    }

    pub fn idx_path(&self) -> &Path {
        &self.files.idx_path
    }
}

pub struct IdxWriteFixture {
    _dir: TempDir,
    pub idx_path: PathBuf,
    pub samples: Vec<BenchSample>,
    pub map: Map,
}

impl IdxWriteFixture {
    pub fn new(num_samples: usize) -> Self {
        let dir = tempdir().expect("create mmap idx write fixture directory");
        let idx_path = dir.path().join("samples.idx");
        let map_path = dir.path().join("samples.map");
        let samples = make_samples(num_samples);
        let entries = offset_entries(&samples);
        let map = Map::new(&map_path);

        map.push_batch_with_str(
            entries
                .iter()
                .map(|entry| (entry.line_ref.as_str(), entry.offset, entry.length)),
        );

        Self {
            _dir: dir,
            idx_path,
            samples,
            map,
        }
    }
}

pub struct MapFileFixture {
    _dir: TempDir,
    pub map_path: PathBuf,
    pub entries: Vec<MapEntry>,
}

impl MapFileFixture {
    pub fn new(num_samples: usize) -> Self {
        let dir = tempdir().expect("create mmap map fixture directory");
        let map_path = dir.path().join("samples.map");
        let samples = make_samples(num_samples);
        let entries = offset_entries(&samples);
        let map = Map::new(&map_path);

        map.push_batch_with_str(
            entries
                .iter()
                .map(|entry| (entry.line_ref.as_str(), entry.offset, entry.length)),
        );
        drop(map);

        Self {
            _dir: dir,
            map_path,
            entries,
        }
    }
}

#[derive(Clone)]
pub struct MapEntry {
    pub line_ref: String,
    pub offset: u64,
    pub length: u64,
}

pub struct SampleFiles {
    _dir: TempDir,
    bin_path: PathBuf,
    idx_path: PathBuf,
}

impl SampleFiles {
    fn new(num_samples: usize) -> Self {
        let dir = tempdir().expect("create mmap sample fixture directory");
        let bin_path = dir.path().join("samples.bin");
        let idx_path = dir.path().join("samples.idx");
        let map_path = dir.path().join("samples.map");
        let samples = make_samples(num_samples);
        let offsets = write_text_bin_file(&bin_path, &samples);
        let map = Map::new(&map_path);

        map.push_batch_with_str(
            samples
                .iter()
                .zip(offsets.iter())
                .map(|(sample, offset)| (sample.text.as_str(), offset.0, offset.1)),
        );

        let mut idx_writer =
            rwkv_data::mmap::idx::IdxWriter::<BenchSample>::new(&idx_path, 64 * 1024);
        for sample in &samples {
            idx_writer.push(sample, &map);
        }
        idx_writer.update_metadata();
        drop(idx_writer);
        drop(map);

        Self {
            _dir: dir,
            bin_path,
            idx_path,
        }
    }
}

pub fn make_tokens(num_tokens: usize) -> Vec<u16> {
    (0..num_tokens)
        .map(|index| ((index.wrapping_mul(31).wrapping_add(7)) % 65_521) as u16)
        .collect()
}

pub fn make_samples(num_samples: usize) -> Vec<BenchSample> {
    (0..num_samples)
        .map(|index| BenchSample {
            id: index as u64,
            score: ((index * 17) % 10_003) as u64,
            text: format!(
                "sample-{index:05}-rwkv mmap benchmark text with deterministic bytes {}",
                index % 97
            ),
        })
        .collect()
}

pub fn offset_entries(samples: &[BenchSample]) -> Vec<MapEntry> {
    let mut offset = 0;
    let mut entries = Vec::with_capacity(samples.len());

    for sample in samples {
        let length = sample.text.len() as u64;
        entries.push(MapEntry {
            line_ref: sample.text.clone(),
            offset,
            length,
        });
        offset += length;
    }

    entries
}

pub fn write_bin_file(path: &Path, tokens: &[u16]) {
    let mut writer = BinWriter::<u16>::new(path, 1, 64 * 1024);
    writer.push(tokens);
    writer.update_metadata();
}

pub fn write_text_bin_file(path: &Path, samples: &[BenchSample]) -> Vec<(u64, u64)> {
    let mut writer = BinWriter::<u8>::new(path, 1, 64 * 1024);
    let offsets = samples
        .iter()
        .map(|sample| writer.push(sample.text.as_bytes()))
        .collect::<Vec<_>>();
    writer.update_metadata();
    offsets
}

pub fn write_legacy_bin_files(bin_path: &Path, tokens: &[u16]) {
    let mut idx_path = bin_path.to_path_buf();
    idx_path.set_extension("idx");

    let mut bin_file = File::create(bin_path).expect("create legacy mmap bin fixture");
    for token in tokens {
        bin_file
            .write_all(&token.to_le_bytes())
            .expect("write legacy mmap bin fixture token");
    }

    let num_lines = 4_096usize.min(tokens.len());
    let base_tokens_per_line = tokens.len() / num_lines;
    let extra_tokens = tokens.len() % num_lines;
    let mut idx_file = File::create(idx_path).expect("create legacy mmap idx fixture");

    idx_file
        .write_all(b"MMIDIDX\0\0")
        .expect("write legacy mmap idx header");
    idx_file
        .write_all(&[1, 0, 0, 0, 0, 0, 0, 0])
        .expect("write legacy mmap idx version");
    idx_file
        .write_all(&[8])
        .expect("write legacy mmap idx dtype");
    idx_file
        .write_all(&(num_lines as u64).to_le_bytes())
        .expect("write legacy mmap idx line count");
    idx_file
        .write_all(&0u64.to_le_bytes())
        .expect("write legacy mmap idx boundary count");

    for line_index in 0..num_lines {
        let num_line_tokens = base_tokens_per_line + usize::from(line_index < extra_tokens);
        idx_file
            .write_all(&(num_line_tokens as u32).to_le_bytes())
            .expect("write legacy mmap idx token count");
    }
}
