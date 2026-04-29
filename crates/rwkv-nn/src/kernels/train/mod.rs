pub mod channel_mixer;
pub mod lm_head_l2wrap_ce;
pub mod time_mixer;

/// We create our own Backend trait that extends the Burn backend trait.
pub trait TrainBackend: burn::tensor::backend::Backend {}

impl<B> TrainBackend for B where B: burn::tensor::backend::Backend {}

/// We create our own AutodiffBackend trait that extends the Burn autodiff backend trait.
pub trait AutodiffBackend: TrainBackend + burn::tensor::backend::AutodiffBackend {}

impl<B> AutodiffBackend for B where B: TrainBackend + burn::tensor::backend::AutodiffBackend {}
