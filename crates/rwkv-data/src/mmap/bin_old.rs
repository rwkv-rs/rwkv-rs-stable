//! Legacy mmap `.bin/.idx` token-stream reader.
//!
//! The legacy `.idx` sidecar starts with `MMIDIDX\0\0`, an eight-byte version
//! field `[1, 0, 0, 0, 0, 0, 0, 0]`, a dtype byte, the line count, an unused
//! boundary count, and one `u32` token count per line. The `.bin` file stores
//! contiguous `u16` token units with no modern 92-byte header. Public offsets
//! and lengths are expressed in logical tokens.

use std::{borrow::Cow, fs::File, io::Read, marker::PhantomData, mem::size_of, path::Path};

use bytemuck::try_cast_slice;
use log::info;
use memmap2::{Mmap, MmapOptions};
use primes::is_prime;

use crate::mmap::dtype::{TokenUnit, TokenUnitDType};

const MMAP_MAGIC_HDR: &[u8] = b"MMIDIDX\x00\x00";
const RWKV_MMAP_VERSION: [u8; 8] = [1, 0, 0, 0, 0, 0, 0, 0];
const RWKV_MMAP_DTYPE_U16: u8 = 8;

/// 旧版 RWKV `.bin/.idx` 读取器，实现与新版 `BinReader` 对齐的接口。
/// Legacy RWKV `.bin/.idx` reader reusing the modern `BinReader` interface.
pub struct BinReader<T: TokenUnit> {
    /// Number of logical tokens computed from the legacy `.idx` sidecar.
    pub num_tokens: u64,
    /// Number of serialized units that make up one logical token.
    pub num_units_per_token: u64,
    /// Token unit dtype from the legacy `.idx` sidecar.
    pub dtype: TokenUnitDType,
    mmap: Mmap,
    metadata: Metadata,
    _phantom: PhantomData<T>,
}

impl<T: TokenUnit> BinReader<T> {
    /// 构建旧版 mmap 读取器；自动根据 `.bin` 路径定位 `.idx`。
    /// Build legacy mmap reader; derive the companion `.idx` from the `.bin` path.
    ///
    /// # Panics
    ///
    /// Panics if the `.bin` file cannot be opened or memory-mapped, the
    /// companion `.idx` file cannot be opened or read, the legacy header or
    /// version is invalid, the dtype is unsupported or does not match `T`, the
    /// token-unit size is not `u16`, expected file size calculation overflows,
    /// or the `.bin` file size does not match `.idx` metadata.
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        let bin_path = path.as_ref();
        let mut idx_path = bin_path.to_path_buf();
        idx_path.set_extension("idx");

        let bin_file = File::open(bin_path)
            .unwrap_or_else(|err| panic!("Failed to open .bin file {bin_path:?}: {err}"));
        let mmap = unsafe {
            MmapOptions::new()
                .map(&bin_file)
                .unwrap_or_else(|err| panic!("Failed to mmap .bin file {bin_path:?}: {err}"))
        };

        info!("Reading legacy mmap metadata from {idx_path:?}.");
        let token_unit_size = size_of::<T>();
        let metadata = read_metadata(&idx_path, token_unit_size);
        let num_tokens = metadata.num_tokens as u64;
        let num_units_per_token = metadata.num_units_per_token as u64;
        let dtype = metadata.dtype;
        assert_eq!(
            dtype,
            T::DTYPE,
            "Token type mismatch: metadata dtype {:?} but reader requested {:?}.",
            dtype,
            T::DTYPE
        );

        info!(
            "Metadata parsed successfully (num_tokens={}, num_units_per_token={}).",
            num_tokens, num_units_per_token
        );

        let expected_mmap_len =
            calculate_expected_len(&metadata, token_unit_size).unwrap_or_else(|| {
                panic!(
                    "Overflow calculating expected size for .bin file {:?}.",
                    bin_path
                )
            });

        assert_eq!(
            mmap.len(),
            expected_mmap_len,
            "Mismatched .bin file size. Expected {} ({} units * {} bytes/unit), got {}. Path: {:?}",
            expected_mmap_len,
            metadata.num_token_units,
            token_unit_size,
            mmap.len(),
            bin_path
        );

        info!("Legacy mmap BinReader created.");

        Self {
            mmap,
            metadata,
            num_tokens,
            num_units_per_token,
            dtype,
            _phantom: PhantomData,
        }
    }

    /// 读取 magic_prime，保持与新版读取器一致的检测逻辑。
    /// Derive `magic_prime` matching the new reader's validation rules.
    ///
    /// # Panics
    ///
    /// Panics if `ctx_len` is zero, the dataset is too small for the context
    /// length, no valid prime exists in the computed slot range, or the chosen
    /// prime is outside the expected dataset-size range.
    pub fn get_magic_prime(&self, ctx_len: u64) -> u64 {
        if ctx_len == 0 {
            panic!("Context length must be positive.");
        }
        let ctx_len_usize = ctx_len as usize;
        ensure_dataset_scale(self.metadata.num_tokens, ctx_len_usize);

        let dataset_slot = (self.metadata.num_tokens / ctx_len_usize) as u64;

        info!(
            "Searching magic_prime within dataset slots: dataset_slot = {}",
            dataset_slot
        );

        let magic_prime = (0..dataset_slot)
            .rev()
            .find(|value| value % 3 == 2 && is_prime(*value))
            .unwrap_or_else(|| panic!("Failed to locate a valid magic_prime in dataset slots."));

        let ratio = magic_prime as f64 / dataset_slot as f64;
        assert!(
            ratio > 0.9 && ratio <= 1.0,
            "Invalid magic_prime {} found for dataset slot {}.",
            magic_prime,
            dataset_slot
        );

        magic_prime
    }

    /// 提供给定偏移的 Token 窗口，若越界则自动回绕。
    /// Fetch a token window with wraparound semantics.
    ///
    /// If the requested window reaches the end of the token stream, reading
    /// wraps around to the first token and returns an owned buffer. Non-wrapped
    /// reads borrow directly from the mmap.
    ///
    /// # Panics
    ///
    /// Panics if offset or length arithmetic overflows after conversion to
    /// `usize`, if unchecked mmap slice bounds are outside the file, or if the
    /// selected bytes cannot be cast to `T` token units.
    pub fn get(&'_ self, offset: u64, length: u64) -> Cow<'_, [T]> {
        let unit_size = self.dtype.get_token_unit_size();
        let unit_offset = (offset * self.num_units_per_token) as usize;
        let unit_length = (length * self.num_units_per_token) as usize;
        let byte_offset = unit_offset * unit_size;
        let byte_length = unit_length * unit_size;

        if offset + length <= self.num_tokens {
            let byte_slice = &self.mmap[byte_offset..byte_offset + byte_length];
            Cow::from(try_cast_slice(byte_slice).expect("Failed to cast slice to TokenUnit array."))
        } else {
            let mut result = Vec::with_capacity(unit_length);
            let first_part_len = self.metadata.num_token_units - unit_offset;
            let first_byte_slice = &self.mmap[byte_offset..];

            result.extend_from_slice(
                try_cast_slice(first_byte_slice)
                    .expect("Failed to cast first part of wrapped slice."),
            );

            let second_part_len = unit_length - first_part_len;
            let second_byte_slice = &self.mmap[..second_part_len * unit_size];

            result.extend_from_slice(
                try_cast_slice(second_byte_slice)
                    .expect("Failed to cast second part of wrapped slice."),
            );

            Cow::from(result)
        }
    }
}

fn calculate_expected_len(metadata: &Metadata, unit_size: usize) -> Option<usize> {
    metadata.num_token_units.checked_mul(unit_size)
}

fn ensure_dataset_scale(num_tokens: usize, context_length: usize) {
    assert!(
        num_tokens > context_length * 3,
        "Dataset is too small. Please increase the data."
    );
}

/// `.idx` 元信息，描述数据集的结构。
/// Metadata extracted from `.idx`, describing dataset structure.
#[derive(Clone, Debug)]
pub struct Metadata {
    /// Token unit dtype declared by the legacy sidecar.
    pub dtype: TokenUnitDType,
    /// Number of logical input lines recorded in the sidecar.
    pub num_lines: usize,
    /// Total number of logical tokens across all lines.
    pub num_tokens: usize,
    /// Total number of serialized token units in the `.bin` file.
    pub num_token_units: usize,
    /// Number of serialized units that make up one logical token.
    pub num_units_per_token: usize,
}

/// 解析 `.idx` 文件并返回基础元数据。
/// Parse the `.idx` file and return foundational metadata.
fn read_metadata(idx_path: &Path, token_unit_size: usize) -> Metadata {
    let mut idx_file =
        File::open(idx_path).unwrap_or_else(|_| panic!("Failed to open .idx file {:?}", idx_path));

    let magic_hdr = read_exact::<9>(&mut idx_file, "magic_hdr");
    let version = read_exact::<8>(&mut idx_file, "version");

    if magic_hdr != MMAP_MAGIC_HDR || version != RWKV_MMAP_VERSION {
        panic!(
            "Unknown .idx file format. Magic: {:?}, Version: {:?}. Path: {:?}",
            magic_hdr, version, idx_path
        );
    }

    let dtype_code = read_exact::<1>(&mut idx_file, "dtype_code");

    if token_unit_size != size_of::<u16>() {
        panic!(
            "RWKV .idx format expects TokenUnit to be u16 (size {}), but TokenUnit size is {}.",
            size_of::<u16>(),
            token_unit_size
        );
    }
    if dtype_code[0] != RWKV_MMAP_DTYPE_U16 {
        panic!(
            "RWKV .idx file DTYPE mismatch. Expected code {}, got {}.",
            RWKV_MMAP_DTYPE_U16, dtype_code[0]
        );
    }

    let num_lines = u64::from_le_bytes(read_exact::<8>(&mut idx_file, "num_lines")) as usize;

    let _num_boundaries = read_exact::<8>(&mut idx_file, "num_boundaries");

    let mut buffer = vec![0; num_lines * 4];
    idx_file
        .read_exact(&mut buffer)
        .expect("Failed to read token counts from idx file");
    let nums_tokens_per_line: Vec<u32> = buffer
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    info!(
        "Loaded {} per-line token counts from idx.",
        nums_tokens_per_line.len()
    );

    let num_tokens = nums_tokens_per_line
        .iter()
        .map(|&x| x as usize)
        .sum::<usize>();
    info!("Total token count computed: {}", num_tokens);

    Metadata {
        dtype: TokenUnitDType::U16,
        num_lines,
        num_tokens,
        num_token_units: num_tokens,
        num_units_per_token: 1,
    }
}

/// 从文件中读取固定长度的字节数组。
/// Read a fixed-length byte block from file.
fn read_exact<const N: usize>(file: &mut File, what: &str) -> [u8; N] {
    let mut buf = [0u8; N];
    file.read_exact(&mut buf)
        .unwrap_or_else(|_| panic!("读取idx文件失败: {}", what));
    buf
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::Write};

    use serde::Deserialize;
    use tempfile::tempdir;

    use super::{BinReader, MMAP_MAGIC_HDR, RWKV_MMAP_DTYPE_U16, RWKV_MMAP_VERSION};
    use crate::mmap::dtype::TokenUnitDType;

    #[derive(Debug, Deserialize)]
    struct LegacyJsonlSample {
        tokens: Vec<u16>,
    }

    #[test]
    fn bin_reader_get() {
        let jsonl = [
            r#"{"tokens":[1,2,3]}"#,
            r#"{"tokens":[4,5]}"#,
            r#"{"tokens":[6,7,8,9]}"#,
        ];
        let samples = jsonl
            .iter()
            .map(|line| sonic_rs::from_str::<LegacyJsonlSample>(line).unwrap())
            .collect::<Vec<_>>();

        let dir = tempdir().unwrap();
        let bin_path = dir.path().join("legacy.bin");
        let mut idx_path = bin_path.clone();
        idx_path.set_extension("idx");

        let mut bin_file = File::create(&bin_path).unwrap();
        for token in samples.iter().flat_map(|sample| &sample.tokens) {
            bin_file.write_all(&token.to_le_bytes()).unwrap();
        }

        let mut idx_file = File::create(&idx_path).unwrap();
        idx_file.write_all(MMAP_MAGIC_HDR).unwrap();
        idx_file.write_all(&RWKV_MMAP_VERSION).unwrap();
        idx_file.write_all(&[RWKV_MMAP_DTYPE_U16]).unwrap();
        idx_file
            .write_all(&(samples.len() as u64).to_le_bytes())
            .unwrap();
        idx_file.write_all(&0u64.to_le_bytes()).unwrap();

        for sample in &samples {
            idx_file
                .write_all(&(sample.tokens.len() as u32).to_le_bytes())
                .unwrap();
        }

        let reader = BinReader::<u16>::new(&bin_path);
        let tokens = samples
            .iter()
            .flat_map(|sample| sample.tokens.iter().copied())
            .collect::<Vec<_>>();

        assert_eq!(reader.num_tokens, tokens.len() as u64);
        assert_eq!(reader.num_units_per_token, 1);
        assert_eq!(reader.dtype, TokenUnitDType::U16);
        assert_eq!(reader.get(0, tokens.len() as u64).as_ref(), tokens);
        assert_eq!(reader.get(3, 2).as_ref(), &[4, 5]);
        assert_eq!(reader.get(7, 4).as_ref(), &[8, 9, 1, 2]);
    }
}
