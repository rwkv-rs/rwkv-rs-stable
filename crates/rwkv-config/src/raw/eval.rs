use serde::{Deserialize, Serialize};

use crate::fill_default;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
/// User-provided evaluation schema loaded from TOML before validation.
pub struct RawEvalConfig {
    /// Experiment name provided by the TOML file.
    pub experiment_name: String,
    /// Experiment description provided by the TOML file.
    pub experiment_desc: String,
    /// Optional TOML admin API key.
    pub admin_api_key: Option<String>,
    /// Optional TOML run mode; defaults to `new` when omitted.
    pub run_mode: Option<String>,
    /// Optional TOML checker skip switch; defaults to `false` when omitted.
    pub skip_checker: Option<bool>,
    /// Optional TOML dataset-check skip switch; defaults to `false` when omitted.
    pub skip_dataset_check: Option<bool>,
    /// Optional TOML judger concurrency; defaults to `8` when omitted.
    pub judger_concurrency: Option<usize>,
    /// Optional TOML checker concurrency; defaults to `8` when omitted.
    pub checker_concurrency: Option<usize>,
    /// Optional TOML database pool size; defaults to `32` when omitted.
    pub db_pool_max_connections: Option<u32>,
    /// Model architecture versions provided by the TOML file.
    pub model_arch_versions: Vec<String>,
    /// Model data versions provided by the TOML file.
    pub model_data_versions: Vec<String>,
    /// Model parameter-size labels provided by the TOML file.
    pub model_num_params: Vec<String>,
    /// Benchmark fields provided by the TOML file.
    pub benchmark_field: Vec<String>,
    /// Extra benchmark names provided by the TOML file.
    pub extra_benchmark_name: Vec<String>,
    /// Optional TOML Space upload switch; defaults to `false` when omitted.
    pub upload_to_space: Option<bool>,
    /// Git hash provided by the TOML file.
    pub git_hash: String,
    /// Internal API model entries provided by the TOML file.
    pub models: Vec<IntApiConfig>,
    /// External LLM judger API config provided by the TOML file.
    pub llm_judger: ExtApiConfig,
    /// External LLM checker API config provided by the TOML file.
    pub llm_checker: ExtApiConfig,
    /// Space database config provided by the TOML file.
    pub space_db: SpaceDbConfig,
}

impl RawEvalConfig {
    /// Fills omitted optional evaluation TOML values with raw-schema defaults.
    pub fn fill_default(&mut self) {
        fill_default!(
            self,
            upload_to_space: false,
            run_mode: "new".to_string(),
            skip_checker: false,
            skip_dataset_check: false,
            judger_concurrency: 8,
            checker_concurrency: 8,
            db_pool_max_connections: 32
        );
        self.space_db.fill_default();
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
/// User-provided internal evaluation API schema loaded from TOML.
pub struct IntApiConfig {
    /// Model architecture version provided by the TOML file.
    pub model_arch_version: String,
    /// Model data version provided by the TOML file.
    pub model_data_version: String,
    /// Model parameter-size label provided by the TOML file.
    pub model_num_params: String,
    /// Base URL provided by the TOML file.
    pub base_url: String,
    /// API key provided by the TOML file.
    pub api_key: String,
    /// Served model identifier provided by the TOML file.
    pub model: String,
    /// Optional TOML maximum batch size for this internal model.
    pub max_batch_size: Option<usize>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
/// User-provided external evaluation API schema loaded from TOML.
pub struct ExtApiConfig {
    /// Base URL provided by the TOML file.
    pub base_url: String,
    /// API key provided by the TOML file.
    pub api_key: String,
    /// Served model identifier provided by the TOML file.
    pub model: String,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
/// User-provided Hugging Face Space database schema loaded from TOML.
pub struct SpaceDbConfig {
    /// Database username provided by the TOML file.
    pub username: String,
    /// Database password provided by the TOML file.
    pub password: String,
    /// Database host provided by the TOML file.
    pub host: String,
    /// Database port provided by the TOML file.
    pub port: String,
    /// Database name provided by the TOML file.
    pub database_name: String,
    /// Optional TOML SSL mode; defaults to `verify-full` when omitted.
    pub sslmode: Option<String>,
}

impl SpaceDbConfig {
    /// Fills omitted optional Space database TOML values with raw-schema defaults.
    pub fn fill_default(&mut self) {
        fill_default!(self, sslmode: "verify-full".to_string());
    }
}
