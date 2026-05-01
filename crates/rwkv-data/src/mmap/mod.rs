//! Memory-mapped dataset storage for token streams and line-reference samples.
//!
//! The mmap dataset format is split across token `.bin` files, sample `.idx`
//! files, and a redb-backed `.map` lookup from stable line references to token
//! offsets and lengths.

/// Modern RWKV token-stream `.bin` reader and writer.
pub mod bin;
/// Legacy RWKV `.bin/.idx` reader for old mmap exports.
pub mod bin_old;
/// Token unit traits and dtype metadata used by mmap readers and writers.
pub mod dtype;
/// Sample index `.idx` reader and writer.
pub mod idx;
/// Line-reference map from sample text hashes to token ranges.
pub mod map;
/// Deterministic sample offset scheduling helpers.
pub mod sample;

const MMAP_VERSION: [u8; 8] = 1u64.to_le_bytes();

#[cfg(test)]
mod tests {
    use rwkv_derive::LineRef;
    use serde::{Deserialize, Serialize};
    use tempfile::tempdir;

    use crate as rwkv_data;
    use super::{
        bin::{BinReader, BinWriter},
        idx::{IdxReader, IdxWriter},
        map::Map,
    };

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, LineRef)]
    struct JsonlSample {
        id: u64,
        score: u64,
        #[line_ref]
        text: String,
    }

    #[test]
    fn jsonl_mmap_roundtrip() {
        let jsonl = [
            r#"{"id":1,"score":10,"text":"rwkv"}"#,
            r#"{"id":2,"score":20,"text":"中文样本"}"#,
            r#"{"id":3,"score":30,"text":"line with spaces and punctuation, ok."}"#,
        ];
        let samples = jsonl
            .iter()
            .map(|line| sonic_rs::from_str::<JsonlSample>(line).unwrap())
            .collect::<Vec<_>>();

        let dir = tempdir().unwrap();
        let bin_path = dir.path().join("samples.bin");
        let idx_path = dir.path().join("samples.idx");
        let map_path = dir.path().join("samples.map");

        let mut bin_writer = BinWriter::<u8>::new(&bin_path, 1, 1024);
        let mut map = Map::new(&map_path);
        let mut idx_writer = IdxWriter::<JsonlSample>::new(&idx_path, 1024);

        for sample in &samples {
            let (offset, length) = bin_writer.push(sample.text.as_bytes());
            map.push_with_str(&sample.text, offset, length);
            idx_writer.push(sample, &map);
        }

        bin_writer.update_metadata();
        idx_writer.update_metadata();
        drop(bin_writer);
        drop(idx_writer);

        let bin_reader = BinReader::<u8>::new(&bin_path);
        let idx_reader = IdxReader::<JsonlSample>::new(&idx_path);

        let restored = (0..idx_reader.num_samples)
            .map(|index| idx_reader.get(index, &bin_reader))
            .collect::<Vec<_>>();

        assert_eq!(restored, samples);
    }

    #[test]
    fn bin_reader_get() {
        let dir = tempdir().unwrap();
        let bin_path = dir.path().join("tokens.bin");

        let mut writer = BinWriter::<u8>::new(&bin_path, 1, 1024);
        writer.push(b"abcdef");
        writer.update_metadata();
        drop(writer);

        let reader = BinReader::<u8>::new(&bin_path);

        assert_eq!(reader.get(1, 3).as_ref(), b"bcd");
        assert_eq!(reader.get(4, 4).as_ref(), b"efab");
    }

    #[test]
    fn bin_reader_get_u16() {
        let dir = tempdir().unwrap();
        let bin_path = dir.path().join("tokens.bin");
        let tokens = [10_u16, 20, 30, 40, 50, 60];

        let mut writer = BinWriter::<u16>::new(&bin_path, 1, 1024);
        writer.push(&tokens);
        writer.update_metadata();
        drop(writer);

        let reader = BinReader::<u16>::new(&bin_path);

        assert_eq!(reader.get(1, 3).as_ref(), &[20, 30, 40]);
        assert_eq!(reader.get(4, 4).as_ref(), &[50, 60, 10, 20]);
    }
}
