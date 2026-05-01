//! Sample index `.idx` files for mmap datasets.
//!
//! An index file starts with a 40-byte metadata header: bytes 0..8 are
//! `RWKVIDX\0`, bytes 8..16 are the mmap version, bytes 16..24 store the sample
//! count, bytes 24..32 store the serialized sample size, and bytes 32..40 store
//! a magic prime. Fixed-size serialized samples follow immediately after the
//! header.

use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Seek, SeekFrom, Write},
    marker::PhantomData,
    mem::size_of,
    path::Path,
};

use bytemuck::{Pod, Zeroable, bytes_of, from_bytes};
use memmap2::{Mmap, MmapOptions};

use crate::mmap::{MMAP_VERSION, bin::BinReader, map::Map, sample::calculate_magic_prime};

/// Sample type that can be stored in a fixed-size mmap `.idx` record.
pub trait LineRefSample {
    /// Plain-old-data record written to and read from the `.idx` file.
    type Serialized: Pod + Zeroable;

    /// Converts a sample into its fixed-size record.
    ///
    /// Implementations typically resolve variable-length text through `map`
    /// and store token `(offset, length)` pairs in the record.
    fn to_serialized(&self, map: &Map) -> Self::Serialized;

    /// Reconstructs a sample from a fixed-size record and token `.bin` reader.
    fn from_serialized(data: &Self::Serialized, bin: &BinReader<u8>) -> Self;
}

/// Reader for fixed-size sample index `.idx` files.
pub struct IdxReader<T: LineRefSample> {
    /// Number of samples recorded in the metadata header.
    pub num_samples: u64,
    /// Size in bytes of each serialized sample record.
    pub sample_size: u64,
    reader: Mmap,
    _phantom: PhantomData<T>,
}

impl<T: LineRefSample> IdxReader<T> {
    /// Opens and validates a sample index `.idx` file.
    ///
    /// # Panics
    ///
    /// Panics if the file cannot be opened or memory-mapped, the header or
    /// version is invalid, metadata bytes cannot be decoded, `sample_size` does
    /// not match `T::Serialized`, or the file size does not match the metadata.
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        let file = File::open(&path).unwrap();

        Self {
            num_samples: 0,
            sample_size: 0,
            reader: unsafe { MmapOptions::new().map(&file).unwrap() },
            _phantom: PhantomData,
        }
        .init_metadata()
    }

    fn init_metadata(mut self) -> Self {
        assert_eq!(
            &self.reader[0..8],
            b"RWKVIDX\0",
            "Invalid file header. Are the idx file you provided exported by rwkv-rs."
        );

        assert_eq!(&self.reader[8..16], &MMAP_VERSION, "Unsupported version.");

        self.num_samples = u64::from_le_bytes(self.reader[16..24].try_into().unwrap());
        self.sample_size = u64::from_le_bytes(self.reader[24..32].try_into().unwrap());

        assert_eq!(
            self.sample_size,
            size_of::<T::Serialized>() as u64,
            "IdxReader::init_metadata: serialized size mismatch with type."
        );

        let expected_size = self.num_samples * self.sample_size + 40;

        assert_eq!(
            expected_size,
            self.reader.len() as u64,
            "The file size does not match the metadata. Please check the file integrity."
        );

        self
    }

    /// Returns the magic prime stored in the index header.
    ///
    /// # Panics
    ///
    /// Panics if the magic-prime bytes cannot be read from the mmap header.
    pub fn get_magic_prime(&self) -> u64 {
        u64::from_le_bytes(
            self.reader[32..40]
                .try_into()
                .expect("Failed to read magic prime."),
        )
    }

    /// Reads and reconstructs the sample at `index`.
    ///
    /// # Panics
    ///
    /// Panics if the computed record bounds are outside the mmap, if the record
    /// bytes cannot be interpreted as `T::Serialized`, or if
    /// [`LineRefSample::from_serialized`] panics while reading token ranges.
    pub fn get(&self, index: u64, bin: &BinReader<u8>) -> T {
        let byte_offset = (index * self.sample_size + 40) as usize;
        let byte_length = self.sample_size as usize;

        let byte_slice = &self.reader[byte_offset..byte_offset + byte_length];
        let serialized: &T::Serialized = from_bytes(byte_slice);

        T::from_serialized(serialized, bin)
    }
}

/// Writer for fixed-size sample index `.idx` files.
///
/// Call [`IdxWriter::update_metadata`] after the final write so readers can
/// validate sample counts and magic-prime metadata.
pub struct IdxWriter<T: LineRefSample> {
    /// Number of samples written so far.
    pub num_samples: u64,
    /// Size in bytes of each serialized sample record.
    pub sample_size: u64,
    buf_writer: BufWriter<File>,
    pos: u64,
    is_metadata_updated: bool,
    _phantom: PhantomData<T>,
}

impl<T: LineRefSample> IdxWriter<T> {
    /// Creates a new sample index `.idx` file and writes an empty header.
    ///
    /// `capacity` is the internal buffered-writer capacity in bytes.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is zero, the file cannot be created or truncated,
    /// the initial seek fails, or writing the header fails.
    pub fn new<P: AsRef<Path>>(path: P, capacity: usize) -> Self {
        if capacity == 0 {
            panic!("IdxWriter::new: capacity must be greater than zero");
        }

        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .unwrap_or_else(|err| {
                panic!(
                    "IdxWriter::new: failed to open {} for writing: {err}",
                    path.as_ref().display()
                )
            });

        Self {
            num_samples: 0,
            sample_size: size_of::<T::Serialized>() as u64,
            buf_writer: BufWriter::with_capacity(capacity, file),
            pos: 0,
            is_metadata_updated: false,
            _phantom: PhantomData,
        }
        .init_metadata()
    }

    fn init_metadata(mut self) -> Self {
        self.buf_seek(0);
        self.buf_write("file_head", b"RWKVIDX\0", 0, 8);
        self.buf_write("mmap_version", &MMAP_VERSION, 8, 16);
        self.buf_write("num_samples", &[0u8; 8], 16, 24);
        self.buf_write("sample_size", &self.sample_size.to_le_bytes(), 24, 32);
        self.buf_write("magic_prime", &[0u8; 8], 32, 40);
        self.is_metadata_updated = true;
        self
    }

    /// Appends one sample record and returns its sample index.
    ///
    /// The sample is serialized through [`LineRefSample::to_serialized`], which
    /// may use `map` to store token `(offset, length)` references. Call
    /// [`IdxWriter::update_metadata`] after all pushes.
    ///
    /// # Panics
    ///
    /// Panics if sample serialization panics, write-position assertions fail,
    /// or writing the sample bytes fails.
    pub fn push(&mut self, sample: &T, map: &Map) -> u64 {
        let offset = self.num_samples;

        let serialized = sample.to_serialized(map);
        let buf = bytes_of(&serialized);

        let byte_offset = offset * self.sample_size + 40;

        self.buf_write("sample", buf, byte_offset, byte_offset + self.sample_size);
        self.is_metadata_updated = false;

        self.num_samples += 1;

        offset
    }

    /// Rewrites header metadata after sample writes are complete.
    ///
    /// This updates the sample count, serialized sample size, and magic prime,
    /// then seeks back to the end of the file.
    ///
    /// # Panics
    ///
    /// Panics if seeking fails, metadata write-position assertions fail, or any
    /// metadata write fails.
    pub fn update_metadata(&mut self) {
        self.buf_seek(16);
        self.buf_write("num_samples", &self.num_samples.to_le_bytes(), 16, 24);
        self.buf_write("sample_size", &self.sample_size.to_le_bytes(), 24, 32);
        self.buf_seek(32);

        let magic_prime = calculate_magic_prime(self.num_samples);
        self.buf_write("magic_prime", &magic_prime.to_le_bytes(), 32, 40);

        self.buf_writer.seek(SeekFrom::End(0)).unwrap();
        self.is_metadata_updated = true;
    }

    #[inline]
    fn buf_write(&mut self, name: &str, buf: &[u8], start: u64, end: u64) {
        let buf_len = buf.len() as u64;
        assert_eq!(buf_len, end - start, "length mismatch when writing {name}.");
        assert_eq!(start, self.pos, "start mismatch when writing {name}.");
        self.buf_writer
            .write_all(buf)
            .map_err(|e| panic!("IdxWriter::buf_write: failed to write {name}: {e}."))
            .unwrap();
        self.pos += buf_len;
        assert_eq!(end, self.pos, "end mismatch when writing {name}.");
    }

    #[inline]
    fn buf_seek(&mut self, start: u64) {
        self.buf_writer
            .seek(SeekFrom::Start(start))
            .unwrap_or_else(|err| panic!("IdxWriter::buf_seek: failed to seek: {err}"));
        self.pos = start;
    }
}

impl<T: LineRefSample> Drop for IdxWriter<T> {
    fn drop(&mut self) {
        if !self.is_metadata_updated {
            eprintln!(
                "Warning: IdxWriter dropped without updating metadata. So the generated idx file \
                 cannot be read properly."
            );
        }
    }
}
