use serde::{Deserialize, Serialize};

use crate::{DatasetFormatOptions, OptimizerOptions, fill_default};

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
/// User-provided training schema loaded from TOML before validation.
pub struct RawTrainConfig {
    #[serde(alias = "model_cfg_path")]
    /// Model config path provided by `model_cfg` or the legacy `model_cfg_path` TOML key.
    pub model_cfg: String,
    /// Optional TOML value; defaults to `logs` when omitted.
    pub experiment_log_base_path: Option<String>,
    /// Experiment name provided by the TOML file.
    pub experiment_name: String,
    /// Optional TOML value for resuming from an existing record file.
    pub record_path: Option<String>,
    /// Optional TOML value; defaults to `42` when omitted.
    pub random_seed: Option<u64>,
    /// Optional TOML value; defaults to `1` when omitted.
    pub save_freq: Option<usize>,

    /// Dataset directory provided by the TOML file.
    pub dataset_base_path: String,
    /// Dataset filename stem provided by the TOML file.
    pub filename_without_extensions: String,
    /// Optional TOML value; defaults to `DatasetFormatOptions::Rwkv` when omitted.
    pub dataset_format: Option<DatasetFormatOptions>,

    /// Optional TOML value; defaults to `1` when omitted.
    pub num_nodes: Option<usize>,
    /// Optional TOML value; defaults to `1` when omitted.
    pub num_devices_per_node: Option<usize>,
    /// Optional TOML value; defaults to `1` when omitted.
    pub batch_size_per_device: Option<usize>,
    /// Optional TOML value; defaults to `true` when omitted.
    pub grad_checkpoint: Option<bool>,

    /// Optional TOML value used when dataset-derived epoch counts are calculated.
    pub num_dataset_repeats: Option<usize>,
    /// Optional TOML value; defaults to `512` when omitted.
    pub context_length: Option<usize>,
    /// Optional TOML value; defaults to `512` when omitted.
    pub paragraph_length: Option<usize>,

    /// Optimizer selected by the TOML file.
    pub optimizer: OptimizerOptions,
    /// Initial learning rate provided by the TOML file.
    pub learning_rate_start: f32,
    /// Final learning rate provided by the TOML file.
    pub learning_rate_end: f32,
    /// Warmup step count provided by the TOML file.
    pub warmup_steps: usize,
    /// Optional TOML value; defaults to `1e-3` when omitted.
    pub weight_decay: Option<f32>,
    /// Optional TOML value; defaults to `1.0` when omitted.
    pub gradient_clip_val: Option<f32>,
    /// Optional TOML value; defaults to `1` when omitted.
    pub num_accumulation_steps_per_device: Option<usize>,
    /// Optional TOML value; defaults to `true` when omitted.
    pub enable_l2wrap: Option<bool>,

    /// Optional TOML log level; defaults to `warn` when omitted.
    pub level: Option<String>,
    /// Optional TOML value; defaults to `true` when omitted.
    pub use_tui: Option<bool>,
    /// Optional TOML value; defaults to `false` when omitted.
    pub upload_to_wandb: Option<bool>,
    #[serde(skip_serializing)]
    /// Optional TOML WandB API key used only when WandB upload is enabled.
    pub wandb_api_key: Option<String>,
    /// Optional TOML WandB entity name.
    pub wandb_entity_name: Option<String>,
    /// Optional TOML WandB project name used when WandB upload is enabled.
    pub wandb_project_name: Option<String>,
}

impl RawTrainConfig {
    /// Fills omitted optional training TOML values with their raw-schema defaults.
    ///
    /// # Panics
    ///
    /// Panics when `upload_to_wandb` is enabled without both `wandb_api_key`
    /// and `wandb_project_name`.
    pub fn fill_default(&mut self) {
        fill_default!(self,
            experiment_log_base_path: "logs".to_string(),
            random_seed: 42,
            save_freq: 1,
            dataset_format: DatasetFormatOptions::Rwkv,
            num_nodes: 1,
            num_devices_per_node: 1,
            batch_size_per_device: 1,
            grad_checkpoint: true,
            context_length: 512,
            paragraph_length: 512,
            weight_decay: 1e-3,
            gradient_clip_val: 1.0,
            num_accumulation_steps_per_device: 1,
            enable_l2wrap: true,
            level: "warn".to_string(),
            use_tui: true,
            upload_to_wandb: false,
        );
        if self.upload_to_wandb.unwrap() {
            assert!(
                self.wandb_api_key.is_some() && self.wandb_project_name.is_some(),
                "Wandb API Key and Project Name is required."
            );
        } else if self.wandb_api_key.is_some()
            || self.wandb_entity_name.is_some()
            || self.wandb_project_name.is_some()
        {
            eprintln!("Warning: upload_to_wandb is false but WandB fields are set.");
        }
    }
}
