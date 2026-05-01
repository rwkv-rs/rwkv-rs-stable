//! Per-item stream steps that filter or format text batches.

/// Stream filters that keep valid text and exclude rejected text.
pub mod filter;
/// Stream formatters that rewrite text while preserving pipeline order.
pub mod formatter;
