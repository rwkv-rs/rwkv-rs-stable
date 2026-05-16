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
/// Reusable neural-network layers.
#[doc(hidden)]
pub mod layers;
/// RWKV model assemblies built from the crate's reusable layers and modules.
pub mod models;
/// Reusable RWKV neural-network modules.
#[doc(hidden)]
pub mod modules;

/// Test backends and devices used by template kernel tests.
#[cfg(test)]
pub mod test_utils;
