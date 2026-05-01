//! Standalone tensor math and initialization helpers.

/// Dominant singular triplet estimation.
pub mod get_singular_triplet;
/// Parameter initialization and RWKV-specific initialization tensors.
pub mod init_weights;
/// Linear interpolation helper.
pub mod lerp;
/// Tensor normalization helper.
pub mod normalize;
/// QR decomposition helper.
pub mod qr;
// pub mod token_shift;
