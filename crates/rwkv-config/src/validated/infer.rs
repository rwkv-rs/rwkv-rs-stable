use std::{collections::HashSet, sync::Arc};

use once_cell::sync::OnceCell;
use rwkv_derive::ConfigBuilder;
use serde::Serialize;

use crate::raw::infer::GenerationConfig;

#[derive(Clone, Debug, Serialize, ConfigBuilder)]
#[config_builder(raw = "crate::raw::infer::RawInferConfig", cell = "INFER_CFG")]
/// Final validated inference server configuration used at runtime.
pub struct FinalInferConfig {
    // HTTP
    /// Final HTTP bind address.
    pub http_bind_addr: String,
    /// Final request body limit in bytes.
    pub request_body_limit_bytes: usize,
    /// Final SSE keep-alive interval in milliseconds.
    pub sse_keep_alive_ms: u64,
    /// Final optional CORS origin allowlist.
    pub allowed_origins: Option<Vec<String>>,
    #[serde(skip_serializing)]
    /// Final optional API key for authenticated requests.
    pub api_key: Option<String>,

    // Multi-model deployment
    /// Final generation model deployment list.
    pub models: Vec<GenerationConfig>,
}

impl FinalInferConfigBuilder {
    /// Checks final inference builder invariants before publishing the config.
    ///
    /// # Panics
    ///
    /// Panics when `models` has not been loaded, no model is configured, a
    /// model name is empty or duplicated, required model paths are empty,
    /// device IDs are empty, or model batch/context limits are missing or zero.
    pub fn check(&self) {
        let models = self.get_models().unwrap();
        assert!(
            !models.is_empty(),
            "infer config requires at least one model"
        );

        let mut names = HashSet::new();
        for model in models {
            assert!(
                !model.model_name.trim().is_empty(),
                "model_name cannot be empty"
            );
            assert!(
                names.insert(model.model_name.clone()),
                "duplicated model_name: {}",
                model.model_name
            );
            assert!(
                !model.model_cfg.trim().is_empty(),
                "model_cfg cannot be empty for model {}",
                model.model_name
            );
            assert!(
                !model.weights_path.trim().is_empty(),
                "weights_path cannot be empty"
            );
            assert!(
                !model.tokenizer_vocab_path.trim().is_empty(),
                "tokenizer_vocab_path cannot be empty"
            );
            assert!(
                !model.device_ids.is_empty(),
                "device_ids cannot be empty for model {}",
                model.model_name
            );
            assert!(
                model.max_batch_size.unwrap() >= 1,
                "max_batch_size must be >= 1 for model {}",
                model.model_name
            );
            assert!(
                model.max_context_len.unwrap() >= 1,
                "max_context_len must be >= 1 for model {}",
                model.model_name
            );
        }
    }
}

/// Global runtime inference configuration cell.
pub static INFER_CFG: OnceCell<Arc<FinalInferConfig>> = OnceCell::new();
