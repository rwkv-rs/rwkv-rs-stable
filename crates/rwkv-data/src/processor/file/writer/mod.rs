//! File writers that consume processor messages and create output files.

/// CSV and TSV writer adapter.
pub mod csv;
/// JSON Lines writer adapter.
pub mod json;
/// Parquet writer adapter.
pub mod parquet;
