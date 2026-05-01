use std::sync::Arc;

use once_cell::sync::OnceCell;
use rwkv_derive::ConfigBuilder;
use serde::Serialize;

use crate::{DatasetFormatOptions, OptimizerOptions, TokenUnitDType};

#[derive(Clone, Debug, Serialize, ConfigBuilder)]
#[config_builder(raw = "crate::raw::train::RawTrainConfig", cell = "TRAIN_CFG")]
/// Final validated training configuration used at runtime.
pub struct FinalTrainConfig {
    /// Final model config path.
    pub model_cfg: String,
    /// Final experiment log base path.
    pub experiment_log_base_path: Option<String>,
    /// Final experiment name.
    pub experiment_name: String,
    /// Final optional record path used for resume or weight initialization.
    pub record_path: Option<String>,
    /// Final random seed.
    pub random_seed: u64,
    /// Final checkpoint save frequency.
    pub save_freq: usize,
    #[skip_raw]
    /// Builder-calculated flag indicating whether initial weights are needed.
    pub need_init_weight_auto: bool,

    /// Final dataset directory.
    pub dataset_base_path: String,
    /// Final dataset filename stem.
    pub filename_without_extensions: String,
    /// Final dataset format.
    pub dataset_format: DatasetFormatOptions,
    #[skip_raw]
    /// Builder-calculated number of memory-mapped dataset tokens.
    pub mmap_num_tokens_auto: usize,
    #[skip_raw]
    /// Builder-calculated number of memory-map units per token.
    pub mmap_num_units_per_token: usize,
    #[skip_raw]
    /// Builder-calculated memory-map token dtype.
    pub mmap_token_dtype_auto: TokenUnitDType,

    /// Final node count.
    pub num_nodes: usize,
    /// Final device count per node.
    pub num_devices_per_node: usize,
    /// Final per-device batch size.
    pub batch_size_per_device: usize,
    #[skip_raw]
    /// Builder-calculated global batch size.
    pub batch_size_auto: usize,
    /// Final gradient checkpointing switch.
    pub grad_checkpoint: bool,

    /// Final dataset repeat count.
    pub num_dataset_repeats: usize,
    /// Final context length.
    pub context_length: usize,
    /// Final paragraph length.
    pub paragraph_length: usize,

    #[skip_raw]
    /// Builder-calculated mini-epoch count.
    pub num_mini_epochs_auto: usize,
    #[skip_raw]
    /// Builder-calculated step count per mini-epoch.
    pub num_steps_per_mini_epoch_auto: usize,
    #[skip_raw]
    /// Builder-calculated magic prime used by dataset iteration.
    pub magic_prime_auto: usize,

    /// Final optimizer.
    pub optimizer: OptimizerOptions,
    /// Final initial learning rate.
    pub learning_rate_start: f32,
    /// Final ending learning rate.
    pub learning_rate_end: f32,
    /// Final warmup step count.
    pub warmup_steps: usize,
    /// Final weight decay.
    pub weight_decay: f32,
    /// Final gradient clipping value.
    pub gradient_clip_val: f32,
    /// Final accumulation step count per device.
    pub num_accumulation_steps_per_device: usize,
    /// Final L2Wrap enable switch.
    pub enable_l2wrap: bool,

    /// Final log level.
    pub level: String,
    /// Final terminal UI enable switch.
    pub use_tui: bool,
    /// Final WandB upload switch.
    pub upload_to_wandb: bool,

    #[serde(skip_serializing)]
    /// Final optional WandB API key.
    pub wandb_api_key: Option<String>,
    /// Final optional WandB entity name.
    pub wandb_entity_name: Option<String>,
    /// Final optional WandB project name.
    pub wandb_project_name: Option<String>,
}

impl FinalTrainConfigBuilder {
    /// Calculates batch-size auto fields after raw training values have been loaded.
    ///
    /// # Panics
    ///
    /// Panics when `num_nodes`, `num_devices_per_node`, or
    /// `batch_size_per_device` has not been loaded into the builder. Also
    /// panics on division by zero when the calculated batch size is zero.
    pub fn fill_auto_after_load(&mut self) {
        let batch_size_auto = self.num_nodes.unwrap()
            * self.num_devices_per_node.unwrap()
            * self.batch_size_per_device.unwrap();
        let num_steps_per_mini_epoch_auto = 40320 / batch_size_auto;

        self.set_batch_size_auto(Some(batch_size_auto))
            .set_num_steps_per_mini_epoch_auto(Some(num_steps_per_mini_epoch_auto));
    }

    /// Applies record-file discovery results to the builder lifecycle.
    pub fn fill_after_read_record_file(&mut self, record_path: Option<String>) {
        self.set_record_path(record_path);
        if self.record_path.is_none() {
            self.set_need_init_weight_auto(Some(true));
        }
    }

    /// Applies dataset metadata read from the binary dataset to auto fields.
    ///
    /// # Panics
    ///
    /// Panics when `num_dataset_repeats` or `context_length` has not been
    /// loaded into the builder, when `context_length` is zero, or when the
    /// memory-map unit count and token dtype discreteness are inconsistent.
    pub fn fill_after_read_bin(
        &mut self,
        mmap_num_tokens_auto: usize,
        mmap_num_units_per_token: usize,
        mmap_token_dtype_auto: TokenUnitDType,
        magic_prime_auto: usize,
    ) {
        let num_mini_epochs_auto = self.num_dataset_repeats.unwrap() * mmap_num_tokens_auto
            / 40320
            / self.context_length.unwrap();

        assert!(
            (mmap_num_units_per_token == 1 && mmap_token_dtype_auto.is_discrete())
                || (mmap_num_units_per_token != 1 && !mmap_token_dtype_auto.is_discrete())
        );

        self.set_mmap_num_tokens_auto(Some(mmap_num_tokens_auto))
            .set_mmap_num_units_per_token(Some(mmap_num_units_per_token))
            .set_mmap_token_dtype_auto(Some(mmap_token_dtype_auto))
            .set_num_mini_epochs_auto(Some(num_mini_epochs_auto))
            .set_magic_prime_auto(Some(magic_prime_auto));
    }

    /// Checks final training builder invariants before publishing the config.
    ///
    /// # Panics
    ///
    /// Panics when required builder fields have not been loaded, when
    /// `model_cfg` is empty, when multi-node training is requested, or when
    /// `paragraph_length` does not divide `context_length`.
    pub fn check(&self) {
        assert!(
            !self.get_model_cfg().unwrap().trim().is_empty(),
            "model_cfg cannot be empty"
        );
        if self.get_num_nodes().unwrap() > 1 {
            panic!("Multiple nodes training are not supported yet");
        }
        assert!(
            self.get_paragraph_length().unwrap() <= self.get_context_length().unwrap()
                && self.get_context_length().unwrap() % self.get_paragraph_length().unwrap() == 0
        );
    }
}

/// Global runtime training configuration cell.
pub static TRAIN_CFG: OnceCell<Arc<FinalTrainConfig>> = OnceCell::new();
