use std::{
    borrow::Cow,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use burn::data::dataset::Dataset;
use rwkv_config::{DatasetFormatOptions, validated::train::TRAIN_CFG};
use rwkv_data::mmap::{
    bin,
    bin_old,
    dtype::{TokenUnit, TokenUnitDType},
    sample::Sampler,
};

use crate::data::EPOCH_INDEX;

/// Reader facade for current and legacy RWKV mmap `.bin` datasets.
pub enum MmapBinReader<T: TokenUnit> {
    /// Current RWKV mmap format.
    Rwkv(bin::BinReader<T>),
    /// Legacy RWKV mmap format.
    RwkvLegacy(bin_old::BinReader<T>),
}

impl<T: TokenUnit> MmapBinReader<T> {
    /// Opens a mmap dataset with the configured format.
    pub fn open<P: AsRef<std::path::Path>>(path: P, format: DatasetFormatOptions) -> Self {
        match format {
            DatasetFormatOptions::Rwkv => Self::Rwkv(bin::BinReader::new(path)),
            DatasetFormatOptions::RwkvLegacy => Self::RwkvLegacy(bin_old::BinReader::new(path)),
        }
    }

    /// Returns the number of tokens in the dataset.
    pub fn num_tokens(&self) -> u64 {
        match self {
            Self::Rwkv(bin) => bin.num_tokens,
            Self::RwkvLegacy(bin) => bin.num_tokens,
        }
    }

    /// Returns the number of mmap units that represent one token.
    pub fn num_units_per_token(&self) -> u64 {
        match self {
            Self::Rwkv(bin) => bin.num_units_per_token,
            Self::RwkvLegacy(bin) => bin.num_units_per_token,
        }
    }

    /// Returns the token-unit dtype recorded by the dataset metadata.
    pub fn dtype(&self) -> TokenUnitDType {
        match self {
            Self::Rwkv(bin) => bin.dtype,
            Self::RwkvLegacy(bin) => bin.dtype,
        }
    }

    /// Returns the magic prime for the requested context length.
    pub fn get_magic_prime(&self, context_length: u64) -> u64 {
        match self {
            Self::Rwkv(bin) => bin.get_magic_prime(context_length),
            Self::RwkvLegacy(bin) => bin.get_magic_prime(context_length),
        }
    }

    /// Reads a token-unit slice from the mmap dataset.
    pub fn get(&self, offset: u64, length: u64) -> Cow<'_, [T]> {
        match self {
            Self::Rwkv(bin) => bin.get(offset, length),
            Self::RwkvLegacy(bin) => bin.get(offset, length),
        }
    }
}

/// Sliding-window dataset over a mmap token stream.
pub struct SlidingDataset<T: TokenUnit> {
    /// Sequence context length used to convert sample offsets to token offsets.
    pub context_length: u64,
    /// Shared mmap reader used by batchers.
    pub bin: Arc<MmapBinReader<T>>,
    /// One sampler per training device.
    pub samplers: Vec<Sampler>,
    /// Mini-epoch cursor retained for compatibility with the previous dataset structure.
    pub mini_epoch_index: Arc<AtomicUsize>,
}

/// One scheduled sliding-window sample.
#[derive(Clone, Copy, Debug)]
pub struct SlidingSample {
    /// Base sample offset before multiplying by context length.
    pub base_offset: u64,
}

impl<T: TokenUnit> SlidingDataset<T> {
    /// Creates a sliding-window dataset with one sampler per device.
    pub fn new(context_length: u64, bin: Arc<MmapBinReader<T>>, samplers: Vec<Sampler>) -> Self {
        Self {
            context_length,
            bin,
            samplers,
            mini_epoch_index: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl<T: TokenUnit> Dataset<SlidingSample> for SlidingDataset<T> {
    fn get(&self, index: usize) -> Option<SlidingSample> {
        assert!(!self.samplers.is_empty());

        let train_cfg = TRAIN_CFG.get().unwrap();
        let num_samples_per_mini_epoch_per_device =
            train_cfg.num_steps_per_mini_epoch_auto * train_cfg.batch_size_per_device;

        assert!(index < num_samples_per_mini_epoch_per_device * self.samplers.len());

        let device_index = index / num_samples_per_mini_epoch_per_device;
        let local_index = index % num_samples_per_mini_epoch_per_device;
        let mini_epoch_index = EPOCH_INDEX.load(Ordering::Relaxed);

        let sampler = &self.samplers[device_index];
        let base_offset = sampler.get_base_offset(local_index as u64, mini_epoch_index);
        Some(SlidingSample { base_offset })
    }

    fn len(&self) -> usize {
        assert!(!self.samplers.is_empty());

        let train_cfg = TRAIN_CFG.get().unwrap();
        train_cfg.num_steps_per_mini_epoch_auto
            * train_cfg.batch_size_per_device
            * self.samplers.len()
    }
}
