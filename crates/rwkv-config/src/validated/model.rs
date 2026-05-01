use std::sync::Arc;

use once_cell::sync::OnceCell;
use rwkv_derive::ConfigBuilder;
use serde::Serialize;

use crate::ModelTypeOptions;

#[derive(Clone, Debug, Serialize, ConfigBuilder)]
#[config_builder(raw = "crate::raw::model::RawModelConfig", cell = "MODEL_CFG")]
/// Final validated model configuration used at runtime.
pub struct FinalModelConfig {
    /// Final model family.
    pub model_type: ModelTypeOptions,

    /// Final number of recurrent cells.
    pub num_cells: usize,
    /// Final vocabulary size.
    pub vocab_size: usize,
    /// Final embedding dimension.
    pub embedded_dim: usize,
    /// Final number of attention heads.
    pub num_heads: usize,
    /// Final channel-mix dimension scale.
    pub channel_mix_dim_scale: usize,
    /// Final dropout probability.
    pub dropout_prob: f64,

    /// Final token-shift enable switch.
    pub with_token_shift: bool,
    /// Final deep-embedding attention switch.
    pub with_deep_embed_att: bool,
    /// Final deep-embedding feed-forward switch.
    pub with_deep_embed_ffn: bool,

    #[skip_raw]
    /// Builder-calculated attention head size.
    pub head_size_auto: usize,
}

impl FinalModelConfigBuilder {
    /// Calculates auto fields after raw model values have been loaded.
    ///
    /// # Panics
    ///
    /// Panics when `embedded_dim` or `num_heads` has not been loaded into the
    /// builder. Also panics on division by zero when `num_heads` is zero.
    pub fn fill_auto_after_load(&mut self) {
        self.set_head_size_auto(Some(
            self.get_embedded_dim().unwrap() / self.get_num_heads().unwrap(),
        ));
    }
}

/// Global runtime model configuration cell.
pub static MODEL_CFG: OnceCell<Arc<FinalModelConfig>> = OnceCell::new();
