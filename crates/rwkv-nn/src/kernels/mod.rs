use crate::kernels::{addcmul::AddcmulBackend, template::TemplateBackend};

/// Fused elementwise add-and-multiply kernels.
pub mod addcmul;
/// Shared input contract checks for fused kernels.
pub mod check;
pub mod template;
// pub mod guided_token_mask;
// pub mod l2wrap;
// pub mod rapid_sample;
// pub mod token_shift_diff;
// pub mod wkv7_common;
// pub mod wkv7_infer;
// pub mod wkv7_pretrain;
// pub mod wkv7_statepass;
// pub mod wkv7_statetune;

/// We create our own Backend trait that extends the Burn backend trait.
pub trait Backend: burn::tensor::backend::Backend + TemplateBackend + AddcmulBackend {}

impl<B> Backend for B where B: burn::tensor::backend::Backend + TemplateBackend + AddcmulBackend {}

/// We create our own AutodiffBackend trait that extends the Burn autodiff backend trait.
pub trait AutodiffBackend: Backend + burn::tensor::backend::AutodiffBackend {}

impl<B> AutodiffBackend for B where B: Backend + burn::tensor::backend::AutodiffBackend {}
