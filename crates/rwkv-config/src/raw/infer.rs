use serde::{Deserialize, Serialize};

use crate::fill_default;

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
/// User-provided inference server schema loaded from TOML before validation.
pub struct RawInferConfig {
    // HTTP
    /// Optional TOML bind address; defaults to `0.0.0.0:8080` when omitted.
    pub http_bind_addr: Option<String>,
    /// Optional TOML request body limit; defaults to 50 MiB when omitted.
    pub request_body_limit_bytes: Option<usize>,
    /// Optional TOML SSE keep-alive interval; defaults to 10 seconds when omitted.
    pub sse_keep_alive_ms: Option<u64>,
    /// Optional TOML CORS origins; `None` leaves origin policy unset.
    pub allowed_origins: Option<Vec<String>>,
    #[serde(skip_serializing)]
    /// Optional TOML API key for authenticated inference requests.
    pub api_key: Option<String>,
    /// Optional TOML IPC configuration; nested defaults are filled when present.
    pub ipc: Option<RawIpcConfig>,

    // Multi-model deployment
    /// Model deployment entries provided by the TOML file.
    pub models: Vec<GenerationConfig>,
}

impl RawInferConfig {
    /// Fills omitted optional inference TOML values with raw-schema defaults.
    pub fn fill_default(&mut self) {
        fill_default!(
            self,
            http_bind_addr: "0.0.0.0:8080".to_string(),
            request_body_limit_bytes: 50 * 1024 * 1024,
            sse_keep_alive_ms: 10_000u64,
        );

        if let Some(ipc) = self.ipc.as_mut() {
            ipc.fill_default();
        }

        for model in self.models.iter_mut() {
            model.fill_default();
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
/// User-provided IPC inference schema loaded from TOML before validation.
pub struct RawIpcConfig {
    /// Optional TOML IPC enable switch; defaults to `false` when omitted.
    pub enabled: Option<bool>,
    /// Optional TOML service name; defaults to `rwkv.infer.openai` when omitted.
    pub service_name: Option<String>,
    /// Optional TOML request byte limit; defaults to 4 MiB when omitted.
    pub max_request_bytes: Option<usize>,
    /// Optional TOML response byte limit; defaults to 4 MiB when omitted.
    pub max_response_bytes: Option<usize>,
    /// Optional TOML inflight request limit; defaults to `128` when omitted.
    pub max_inflight_requests: Option<usize>,
    /// Optional TOML API-key requirement; defaults to `true` when omitted.
    pub require_api_key: Option<bool>,
}

impl RawIpcConfig {
    /// Fills omitted optional IPC TOML values with raw-schema defaults.
    pub fn fill_default(&mut self) {
        fill_default!(
            self,
            enabled: false,
            service_name: "rwkv.infer.openai".to_string(),
            max_request_bytes: 4 * 1024 * 1024usize,
            max_response_bytes: 4 * 1024 * 1024usize,
            max_inflight_requests: 128usize,
            require_api_key: true,
        );
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
/// User-provided generation model schema loaded from TOML before validation.
pub struct GenerationConfig {
    /// Public model name provided by the TOML file.
    pub model_name: String,
    #[serde(alias = "model_cfg_path")]
    /// Model config path provided by `model_cfg` or the legacy `model_cfg_path` TOML key.
    pub model_cfg: String,
    /// Model weights path provided by the TOML file.
    pub weights_path: String,
    /// Tokenizer vocabulary path provided by the TOML file.
    pub tokenizer_vocab_path: String,

    /// Optional TOML device type; defaults to `0` when omitted.
    pub device_type: Option<u16>,
    /// Device IDs provided by the TOML file.
    pub device_ids: Vec<u32>,

    /// Optional TOML maximum batch size; defaults to `4` when omitted.
    pub max_batch_size: Option<usize>,
    /// Optional TOML paragraph length; defaults to `256` when omitted.
    pub paragraph_len: Option<usize>,
    /// Optional TOML maximum context length; defaults to `4096` when omitted.
    pub max_context_len: Option<usize>,
    /// Optional TOML decode-first switch; defaults to `true` when omitted.
    pub decode_first: Option<bool>,
}

impl GenerationConfig {
    /// Fills omitted optional generation TOML values with raw-schema defaults.
    pub fn fill_default(&mut self) {
        fill_default!(
            self,
            device_type: 0u16,
            max_batch_size: 4usize,
            paragraph_len: 256usize,
            max_context_len: 4096usize,
            decode_first: true,
        );
    }
}
