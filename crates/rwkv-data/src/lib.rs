#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]
#![warn(rustdoc::invalid_html_tags)]

//! Data ingestion, preprocessing, memory mapping, and tokenization utilities.

/// Memory-mapped dataset helpers.
pub mod mmap;
/// Asynchronous text preprocessing pipelines.
pub mod processor;
/// RWKV tokenizer implementation.
pub mod tokenizer;
