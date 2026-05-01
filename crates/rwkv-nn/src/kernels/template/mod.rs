//! Reusable fused-kernel templates.

/// Fused add-multiply template kernel.
pub mod addcmul;
/// Fused matrix-multiply, add, and ReLU template kernel.
pub mod matmul_add_relu;
/// Token-shift difference template kernel.
pub mod token_shift_diff;

use crate::kernels::template::{
    addcmul::AddcmulBackend,
    matmul_add_relu::MatmulAddReluBackend,
    token_shift_diff::TokenShiftDiffBackend,
};

/// We create our own Backend trait that extends the Burn backend trait.
pub trait TemplateBackend:
    burn::tensor::backend::Backend + MatmulAddReluBackend + AddcmulBackend + TokenShiftDiffBackend
{
}

impl<B> TemplateBackend for B where
    B: burn::tensor::backend::Backend
        + MatmulAddReluBackend
        + AddcmulBackend
        + TokenShiftDiffBackend
{
}

/// We create our own AutodiffBackend trait that extends the Burn autodiff backend trait.
pub trait AutodiffBackend: TemplateBackend + burn::tensor::backend::AutodiffBackend {}

impl<B> AutodiffBackend for B where B: TemplateBackend + burn::tensor::backend::AutodiffBackend {}
