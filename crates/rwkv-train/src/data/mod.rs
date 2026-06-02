use std::sync::atomic::AtomicU64;

/// Sliding-window mmap dataset support.
pub mod sliding;

/// Current mini-epoch index used by sliding datasets.
pub static EPOCH_INDEX: AtomicU64 = AtomicU64::new(0);
