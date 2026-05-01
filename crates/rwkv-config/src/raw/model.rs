use serde::Deserialize;

use crate::ModelTypeOptions;

#[derive(Clone, Debug, Deserialize)]
/// User-provided model schema loaded from TOML before validation.
pub struct RawModelConfig {
    /// Model family selected by the TOML file.
    pub model_type: ModelTypeOptions,

    /// Number of recurrent cells provided by the TOML file.
    pub num_cells: usize,
    /// Vocabulary size provided by the TOML file.
    pub vocab_size: usize,
    /// Embedding dimension provided by the TOML file.
    pub embedded_dim: usize,
    /// Number of attention heads provided by the TOML file.
    pub num_heads: usize,
    /// Optional TOML value; defaults to `4` when omitted.
    pub channel_mix_dim_scale: Option<usize>,
    /// Dropout probability provided by the TOML file.
    pub dropout_prob: f64,

    /// Optional TOML value; defaults to `true` when omitted.
    pub with_token_shift: Option<bool>,
    /// Optional TOML value; defaults to `false` when omitted.
    pub with_deep_embed_att: Option<bool>,
    /// Optional TOML value; defaults to `false` when omitted.
    pub with_deep_embed_ffn: Option<bool>,
}

impl RawModelConfig {
    /// Fills omitted optional model TOML values with their raw-schema defaults.
    pub fn fill_default(&mut self) {
        if self.channel_mix_dim_scale.is_none() {
            self.channel_mix_dim_scale = Some(4);
        }

        if self.with_token_shift.is_none() {
            self.with_token_shift = Some(true);
        }

        if self.with_deep_embed_att.is_none() {
            self.with_deep_embed_att = Some(false);
        }

        if self.with_deep_embed_ffn.is_none() {
            self.with_deep_embed_ffn = Some(false);
        }
    }
}
