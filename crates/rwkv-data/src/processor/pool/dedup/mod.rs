//! Deduplication steps that exclude previously observed content.

/// Exact whole-sample hash deduplication.
pub mod exact_hash_sample;
/// Exact sentence-level hash deduplication.
pub mod exact_hash_sentence;
