//! File readers that convert source records into processor text batches.

/// CSV and TSV reader adapter.
pub mod csv;
/// JSON Lines reader adapter.
pub mod json;
/// Parquet reader adapter.
pub mod parquet;
