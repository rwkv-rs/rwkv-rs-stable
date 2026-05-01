//! Training-time fused kernel traits and modules.

/// Channel-mixer fused kernel.
pub mod channel_mixer;
/// Language-model head cross-entropy with L2Wrap fused kernel.
pub mod lm_head_l2wrap_ce;
/// Time-mixer fused kernels.
pub mod time_mixer;

use crate::kernels::train::{
    channel_mixer::ChannelMixerBackend,
    time_mixer::{
        learning_rate_gate::LearningRateGateBackend,
        mix6::Mix6Backend,
        value_residual_gate::ValueResidualGateBackend,
        wkv7::Wkv7Backend,
    },
};

/// We create our own Backend trait that extends the Burn backend trait.
pub trait TrainBackend:
    burn::tensor::backend::Backend
    + ChannelMixerBackend
    + LearningRateGateBackend
    + Mix6Backend
    + ValueResidualGateBackend
    + Wkv7Backend
{
}

impl<B> TrainBackend for B where
    B: burn::tensor::backend::Backend
        + ChannelMixerBackend
        + LearningRateGateBackend
        + Mix6Backend
        + ValueResidualGateBackend
        + Wkv7Backend
{
}

/// We create our own AutodiffBackend trait that extends the Burn autodiff backend trait.
pub trait AutodiffBackend: TrainBackend + burn::tensor::backend::AutodiffBackend {}

impl<B> AutodiffBackend for B where B: TrainBackend + burn::tensor::backend::AutodiffBackend {}
