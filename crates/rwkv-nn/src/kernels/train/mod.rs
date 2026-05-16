//! Training-time fused kernel traits and modules.

/// Channel-mixer fused kernel.
pub mod channel_mixer;
/// Layer-normalization fused kernel.
pub mod layer_norm;
pub(crate) mod layout;
/// Language-model head cross-entropy with L2Wrap fused kernel.
pub mod lm_head_l2wrap_ce;
/// Residual-add fused kernel.
pub mod residual_add;
/// Time-mixer fused kernels.
pub mod time_mixer;

use crate::kernels::train::{
    channel_mixer::ChannelMixerBackend,
    layer_norm::LayerNormBackend,
    lm_head_l2wrap_ce::LmHeadL2WrapCeBackend,
    time_mixer::{
        gated_readout_combine::GatedReadoutCombineBackend,
        key_prepare::KeyPrepareBackend,
        learning_rate_gate::LearningRateGateBackend,
        mix6::Mix6Backend,
        value_residual_gate::ValueResidualGateBackend,
        weight_decay_transform::WeightDecayTransformBackend,
        wkv7::Wkv7Backend,
    },
};

/// We create our own Backend trait that extends the Burn backend trait.
pub trait TrainBackend:
    burn::tensor::backend::Backend
    + ChannelMixerBackend
    + LayerNormBackend
    + LmHeadL2WrapCeBackend
    + GatedReadoutCombineBackend
    + KeyPrepareBackend
    + LearningRateGateBackend
    + Mix6Backend
    + ValueResidualGateBackend
    + WeightDecayTransformBackend
    + Wkv7Backend
{
}

impl<B> TrainBackend for B where
    B: burn::tensor::backend::Backend
        + ChannelMixerBackend
        + LayerNormBackend
        + LmHeadL2WrapCeBackend
        + GatedReadoutCombineBackend
        + KeyPrepareBackend
        + LearningRateGateBackend
        + Mix6Backend
        + ValueResidualGateBackend
        + WeightDecayTransformBackend
        + Wkv7Backend
{
}

/// We create our own AutodiffBackend trait that extends the Burn autodiff backend trait.
pub trait AutodiffBackend: TrainBackend + burn::tensor::backend::AutodiffBackend {}

impl<B> AutodiffBackend for B where B: TrainBackend + burn::tensor::backend::AutodiffBackend {}
