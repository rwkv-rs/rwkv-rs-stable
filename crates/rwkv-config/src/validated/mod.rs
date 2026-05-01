//! Validated configuration structures consumed after raw input resolution.

#[cfg(feature = "eval")]
/// Validated evaluation configuration.
pub mod eval;
#[cfg(feature = "infer")]
/// Validated inference configuration.
pub mod infer;
#[cfg(feature = "model")]
/// Validated model configuration.
pub mod model;
#[cfg(feature = "train")]
/// Validated training configuration.
pub mod train;
