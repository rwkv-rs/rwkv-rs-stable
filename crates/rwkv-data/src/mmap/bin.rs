//! Modern mmap token-stream `.bin` files.
//!
//! A file starts with a 92-byte metadata header:
//! bytes 0..8 are `RWKVBIN\0`, bytes 8..16 are the mmap version, bytes 16..24
//! store the token count, bytes 24..32 store units per token, byte 32 stores the
//! token dtype, bytes 33..89 store magic primes for supported context lengths,
//! and bytes 89..92 are padding. Token units follow immediately after the
//! header. Public offsets and lengths are expressed in logical tokens.

use std::{
    borrow::Cow,
    fs::{File, OpenOptions},
    io::{BufWriter, Seek, SeekFrom, Write},
    marker::PhantomData,
    path::Path,
};

use bytemuck::try_cast_slice;
use memmap2::{Mmap, MmapOptions};
use primes::is_prime;

use crate::mmap::{
    MMAP_VERSION,
    dtype::{TokenUnit, TokenUnitDType},
    sample::calculate_magic_prime,
};

const BIN_DATA_OFFSET: usize = 92;

/// Reader for modern RWKV mmap token-stream `.bin` files.
pub struct BinReader<T: TokenUnit> {
    /// Number of logical tokens recorded in the metadata header.
    pub num_tokens: u64,
    /// Number of serialized units that make up one logical token.
    pub num_units_per_token: u64,
    /// Token unit dtype recorded in the metadata header.
    pub dtype: TokenUnitDType,
    reader: Mmap,
    _phantom: PhantomData<T>,
}

impl<T: TokenUnit> BinReader<T> {
    /// Opens and validates a mmap token-stream `.bin` file.
    ///
    /// # Panics
    ///
    /// Panics if the file cannot be opened or memory-mapped, the header or
    /// version is invalid, the dtype does not match `T`, metadata bytes cannot
    /// be decoded, or the file size does not match the metadata layout.
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        let file = File::open(path).unwrap();

        Self {
            num_tokens: 0,
            num_units_per_token: 0,
            dtype: T::DTYPE,
            reader: unsafe { MmapOptions::new().map(&file).unwrap() },
            _phantom: PhantomData,
        }
        .init_metadata()
    }

    fn init_metadata(mut self) -> Self {
        assert_eq!(
            &self.reader[0..8],
            b"RWKVBIN\0",
            "Invalid file header. Are the bin file you provided exported by rwkv-rs."
        );

        assert_eq!(&self.reader[8..16], MMAP_VERSION, "Unsupported version.");

        self.num_tokens = u64::from_le_bytes(self.reader[16..24].try_into().unwrap());
        self.num_units_per_token = u64::from_le_bytes(self.reader[24..32].try_into().unwrap());

        assert_eq!(self.dtype, TokenUnitDType::get_dtype(self.reader[32]));
        assert_eq!(
            self.num_tokens * self.num_units_per_token * self.dtype.get_token_unit_size() as u64
                + BIN_DATA_OFFSET as u64,
            self.reader.len() as u64,
            "The file size does not match the metadata. Please check the file integrity."
        );
        self
    }

    /// Returns the precomputed magic prime for a supported context length.
    ///
    /// Supported context lengths are 128, 256, 512, 1024, 2048, 4096, and 8192.
    ///
    /// # Panics
    ///
    /// Panics if `ctx_len` is unsupported, the stored value is not prime, the
    /// dataset is too small for the requested context length, or the value is
    /// outside the expected dataset-size range.
    pub fn get_magic_prime(&self, ctx_len: u64) -> u64 {
        let range = match ctx_len {
            128 => 33..41,
            256 => 41..49,
            512 => 49..57,
            1024 => 57..65,
            2048 => 65..73,
            4096 => 73..81,
            8192 => 81..89,
            _ => panic!("Invalid context length: {}", ctx_len),
        };

        let magic_prime = u64::from_le_bytes(
            self.reader[range]
                .try_into()
                .expect("Failed to read magic prime"),
        );

        assert!(
            is_prime(magic_prime),
            "Invalid magic_prime {magic_prime} because it is not a prime."
        );

        assert!(self.num_tokens > ctx_len, "This Datasets is too small.");
        let ratio = magic_prime as f64 / (self.num_tokens / ctx_len) as f64;
        assert!(
            ratio > 0.9 && ratio <= 1.0,
            "Invalid magic_prime {magic_prime} because the value appears to be out of a \
             reasonable range."
        );
        magic_prime
    }

    /// Reads `length` logical tokens starting at `offset`.
    ///
    /// If the requested window reaches the end of the token stream, reading
    /// wraps around to the first token and returns an owned buffer. Non-wrapped
    /// reads borrow directly from the mmap.
    ///
    /// # Panics
    ///
    /// Panics if offset or length arithmetic overflows after conversion to
    /// `usize`, if the unchecked mmap slice bounds are outside the file, or if
    /// the selected bytes cannot be cast to `T` token units.
    pub fn get(&'_ self, offset: u64, length: u64) -> Cow<'_, [T]> {
        let unit_size = self.dtype.get_token_unit_size();
        let unit_offset = (offset * self.num_units_per_token) as usize;
        let unit_length = (length * self.num_units_per_token) as usize;
        let byte_offset = BIN_DATA_OFFSET + unit_offset * unit_size;
        let byte_length = unit_length * unit_size;

        if offset + length <= self.num_tokens {
            let byte_slice = &self.reader[byte_offset..byte_offset + byte_length];
            tokens_from_bytes(byte_slice)
        } else {
            let mut result = Vec::with_capacity(unit_length);
            let first_part_len = self.reader.len() - byte_offset;
            let first_byte_slice = &self.reader[byte_offset..];

            let first_tokens = tokens_from_bytes(first_byte_slice);
            result.extend_from_slice(first_tokens.as_ref());

            let second_part_len = byte_length - first_part_len;
            let second_byte_slice =
                &self.reader[BIN_DATA_OFFSET..BIN_DATA_OFFSET + second_part_len];

            let second_tokens = tokens_from_bytes(second_byte_slice);
            result.extend_from_slice(second_tokens.as_ref());

            Cow::from(result)
        }
    }
}

fn tokens_from_bytes<T: TokenUnit>(byte_slice: &[u8]) -> Cow<'_, [T]> {
    Cow::from(try_cast_slice(byte_slice).expect("Failed to cast slice to TokenUnit array."))
}

/// Writer for modern RWKV mmap token-stream `.bin` files.
///
/// Call [`BinWriter::update_metadata`] after the final write so readers can
/// validate token counts and magic-prime metadata.
pub struct BinWriter<T: TokenUnit> {
    /// Number of logical tokens written so far.
    pub num_tokens: u64,
    /// Number of serialized units that make up one logical token.
    pub num_units_per_token: u64,
    /// Token unit dtype written to the metadata header.
    pub dtype: TokenUnitDType,
    buf_writer: BufWriter<File>,
    pos: u64,
    is_metadata_updated: bool,
    _phantom: PhantomData<T>,
}

impl<T: TokenUnit> BinWriter<T> {
    /// Creates a new mmap token-stream `.bin` file and writes an empty header.
    ///
    /// `capacity` is the internal buffered-writer capacity in bytes.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is zero, the file cannot be created or truncated,
    /// the initial seek fails, or writing the header fails.
    pub fn new<P: AsRef<Path>>(path: P, num_units_per_token: u64, capacity: usize) -> Self {
        if capacity == 0 {
            panic!("Bin::new: capacity must be greater than zero");
        }
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .unwrap_or_else(|err| {
                panic!(
                    "Bin::new: failed to open {} for writing: {err}",
                    &path.as_ref().display(),
                )
            });

        Self {
            num_tokens: 0,
            num_units_per_token,
            dtype: T::DTYPE,
            buf_writer: BufWriter::with_capacity(capacity, file),
            pos: 0,
            is_metadata_updated: false,
            _phantom: PhantomData,
        }
        .init_metadata()
    }

    fn init_metadata(mut self) -> Self {
        self.buf_writer.seek(SeekFrom::Start(0)).unwrap();
        self.buf_write("file_head", b"RWKVBIN\0", 0, 8);
        self.buf_write("mmap_version", &MMAP_VERSION, 8, 16);
        self.buf_write("num_tokens", &[0u8; 8], 16, 24);
        self.buf_write("num_units_per_token", &[0u8; 8], 24, 32);
        self.buf_write("dtype", &(T::DTYPE as u8).to_le_bytes(), 32, 33);
        self.buf_write("magic_primes", &[0u8; 56], 33, 89);
        self.buf_write("padding", &[0u8; 3], 89, BIN_DATA_OFFSET as u64);
        self.is_metadata_updated = true;
        self
    }

    /// Appends tokens and returns their `(offset, length)` in logical tokens.
    ///
    /// The returned range is intended for storage in the map or sample index.
    /// [`BinWriter::update_metadata`] must be called after all pushes.
    ///
    /// # Panics
    ///
    /// Panics if `tokens` is empty, if write-position assertions fail, or if
    /// writing token bytes fails.
    pub fn push(&mut self, tokens: &[T]) -> (u64, u64) {
        if tokens.is_empty() {
            panic!("Empty tokens is not supported.");
        }

        let offset = self.num_tokens;
        let length = tokens.len() as u64;

        let byte_offset =
            offset * self.num_units_per_token * self.dtype.get_token_unit_size() as u64
                + BIN_DATA_OFFSET as u64;
        let byte_length =
            length * self.num_units_per_token * self.dtype.get_token_unit_size() as u64;
        let buf = bytemuck::cast_slice(tokens);

        self.buf_write("token", buf, byte_offset, byte_offset + byte_length);
        self.is_metadata_updated = false;

        self.num_tokens += tokens.len() as u64;

        (offset, length)
    }

    /// Rewrites header metadata after token writes are complete.
    ///
    /// This updates the token count, units-per-token value, and magic-prime
    /// table, then seeks back to the end of the file.
    ///
    /// # Panics
    ///
    /// Panics if seeking fails, metadata write-position assertions fail, or any
    /// metadata write fails.
    pub fn update_metadata(&mut self) {
        self.buf_seek(16);
        self.buf_write("num_tokens", &self.num_tokens.to_le_bytes(), 16, 24);
        self.buf_write(
            "num_units_per_token",
            &self.num_units_per_token.to_le_bytes(),
            24,
            32,
        );
        self.buf_seek(33);
        [128, 256, 512, 1024, 2048, 4096, 8192]
            .iter()
            .enumerate()
            .for_each(|(i, ctx_len)| {
                let name = format!("magic_prime_{}", ctx_len);
                let num_slots = self.num_tokens / ctx_len;
                let magic_prime = calculate_magic_prime(num_slots);
                self.buf_write(
                    &name,
                    &magic_prime.to_le_bytes(),
                    (33 + i * 8) as u64,
                    (41 + i * 8) as u64,
                );
            });
        self.buf_writer.seek(SeekFrom::End(0)).unwrap();
        self.is_metadata_updated = true;
    }

    /// 左闭右开区间, end表示下一次写入的位置
    /// 为什么要设计这个封装:
    /// 1. 减少重复的self.buf_writer.write_all(...).unwrap();
    /// 2. 提高可读性, 便于理解写入内容所对应的实际含义
    /// 3. 防御性编程保证写入位置正确, 要求Coder完全掌握正确的start与end
    #[inline]
    fn buf_write(&mut self, name: &str, buf: &[u8], start: u64, end: u64) {
        let buf_len = buf.len() as u64;
        assert_eq!(buf_len, end - start, "length mismatch when writing {name}.");
        assert_eq!(start, self.pos, "start mismatch when writing {name}.");
        self.buf_writer
            .write_all(buf)
            .map_err(|e| panic!("Bin::buf_write: failed to write {name}: {e}."))
            .unwrap();
        self.pos += buf_len;
        assert_eq!(end, self.pos, "End mismatch when writing {name}.");
    }

    #[inline]
    fn buf_seek(&mut self, start: u64) {
        self.buf_writer.seek(SeekFrom::Start(start)).unwrap();
        self.pos = start;
    }
}

impl<T: TokenUnit> Drop for BinWriter<T> {
    fn drop(&mut self) {
        if !self.is_metadata_updated {
            eprintln!(
                "Warning: BinWriter dropped without updating metadata.So The generated binary \
                 file cannot be read properly."
            );
        }
    }
}
