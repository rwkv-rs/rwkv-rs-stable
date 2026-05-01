#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]
#![warn(rustdoc::invalid_html_tags)]

//! Neural-network layers, modules, fused kernels, and tensor helpers for RWKV models.

/// Recurrent RWKV cell compositions.
pub mod cells;
/// Tensor helper functions used by model initialization and forward passes.
pub mod functions;
/// Custom fused kernel contracts and reference wrappers.
pub mod kernels;
mod layers;
mod modules;

/// Test backends and devices used by crate-local tests.
pub mod test_utils;
