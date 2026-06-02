#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]
#![warn(rustdoc::invalid_html_tags)]

//! Training support for RWKV data loading, learner setup, optimization, logging,
//! and metric rendering.

#[macro_use]
extern crate derive_new;

/// Dataset adapters and training sample scheduling.
pub mod data;
/// Learner initialization and training metric output types.
pub mod learner;
/// Logging integrations used by training loops.
pub mod logger;
/// Optimizer configuration and scheduler helpers.
pub mod optim;
/// Metric renderers for training progress.
pub mod renderer;
/// Shared filesystem and record-file utilities.
pub mod utils;
