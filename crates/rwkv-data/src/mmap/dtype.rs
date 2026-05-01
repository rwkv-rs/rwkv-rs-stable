//! Token unit traits for mmap-backed datasets.

use burn::tensor::Element;
use bytemuck::Pod;
pub use rwkv_config::TokenUnitDType;
use serde::Serialize;

/// Token unit type accepted by mmap token readers and writers.
pub trait TokenUnit: IsDiscrete + Clone + Serialize + Pod + Element {
    /// Serialized dtype marker stored in mmap metadata.
    const DTYPE: TokenUnitDType;
}

impl TokenUnit for u8 {
    const DTYPE: TokenUnitDType = TokenUnitDType::U8;
}

impl TokenUnit for u16 {
    const DTYPE: TokenUnitDType = TokenUnitDType::U16;
}

impl TokenUnit for f32 {
    const DTYPE: TokenUnitDType = TokenUnitDType::F32;
}

/// Marks whether a token unit represents a discrete token id.
pub trait IsDiscrete {
    /// `true` for integer token ids and `false` for continuous values.
    const IS_DISCRETE: bool;
}

impl IsDiscrete for u8 {
    const IS_DISCRETE: bool = true;
}

impl IsDiscrete for u16 {
    const IS_DISCRETE: bool = true;
}

impl IsDiscrete for f32 {
    const IS_DISCRETE: bool = false;
}
