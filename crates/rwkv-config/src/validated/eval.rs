use std::sync::Arc;

use once_cell::sync::OnceCell;
use rwkv_derive::ConfigBuilder;
use serde::Serialize;

use crate::raw::eval::{ExtApiConfig, IntApiConfig, SpaceDbConfig};

#[derive(Clone, Debug, Serialize, ConfigBuilder)]
#[config_builder(raw = "crate::raw::eval::RawEvalConfig", cell = "EVAL_CFG")]
/// Final validated evaluation configuration used at runtime.
pub struct FinalEvalConfig {
    /// Final experiment name.
    pub experiment_name: String,
    /// Final experiment description.
    pub experiment_desc: String,
    /// Final optional admin API key.
    pub admin_api_key: Option<String>,
    /// Final run mode.
    pub run_mode: String,
    /// Final checker skip switch.
    pub skip_checker: bool,
    /// Final dataset-check skip switch.
    pub skip_dataset_check: bool,
    /// Final judger concurrency.
    pub judger_concurrency: usize,
    /// Final checker concurrency.
    pub checker_concurrency: usize,
    /// Final database pool size.
    pub db_pool_max_connections: u32,
    /// Final model architecture versions.
    pub model_arch_versions: Vec<String>,
    /// Final model data versions.
    pub model_data_versions: Vec<String>,
    /// Final model parameter-size labels.
    pub model_num_params: Vec<String>,
    /// Final benchmark fields.
    pub benchmark_field: Vec<String>,
    /// Final extra benchmark names.
    pub extra_benchmark_name: Vec<String>,
    /// Final Space upload switch.
    pub upload_to_space: bool,
    /// Final git hash.
    pub git_hash: String,
    /// Final internal API model entries.
    pub models: Vec<IntApiConfig>,
    /// Final external LLM judger API config.
    pub llm_judger: ExtApiConfig,
    /// Final external LLM checker API config.
    pub llm_checker: ExtApiConfig,
    /// Final optional Space database config.
    pub space_db: Option<SpaceDbConfig>,
}

/// Global runtime evaluation configuration cell.
pub static EVAL_CFG: OnceCell<Arc<FinalEvalConfig>> = OnceCell::new();
