//! Stream filters that return [`StepOutcome::Exclude`] for rejected text.
//!
//! [`StepOutcome::Exclude`]: crate::processor::StepOutcome::Exclude

/// Tokenizer-ratio compression filter.
pub mod compression_ratio_tokenizer;
/// Zstd-ratio compression filter.
pub mod compression_ratio_zstd;
/// Unicode script-mix filter.
pub mod language_char_script;
/// Gopher-style repetition filter.
pub mod repetition_gopher;
