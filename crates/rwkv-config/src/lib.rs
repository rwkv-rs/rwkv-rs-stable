#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]
#![warn(rustdoc::invalid_html_tags)]

//! Configuration schemas and loading helpers for RWKV crates.
//!
//! The crate exposes independent feature families:
//! - `model` enables model architecture configuration.
//! - `train` enables training and dataset configuration.
//! - `infer` enables inference service configuration.
//! - `eval` enables evaluation service configuration.
//!
//! Feature families are independent. Enable only the configuration contracts
//! required by the consuming crate or binary.

/// Raw deserializable configuration structures.
pub mod raw;
/// Validated configuration structures with required fields resolved.
pub mod validated;

use std::{
    fs,
    path::{Path, PathBuf},
};

use serde::de::DeserializeOwned;

#[cfg(any(
    feature = "model",
    feature = "train",
    feature = "infer",
    feature = "eval"
))]
#[doc(hidden)]
pub mod config_builder_helpers {
    pub trait IntoBuilderOption<T> {
        fn into_builder_option(self) -> Option<T>;
    }

    impl<T> IntoBuilderOption<T> for T {
        fn into_builder_option(self) -> Option<T> {
            Some(self)
        }
    }

    impl<T> IntoBuilderOption<T> for Option<T> {
        fn into_builder_option(self) -> Option<T> {
            self
        }
    }
}

/// Loads and deserializes a TOML file into the requested configuration type.
///
/// # Panics
///
/// Panics if the file cannot be read or if the TOML content cannot be
/// deserialized into `T`.
pub fn load_toml<P: AsRef<Path>, T: DeserializeOwned + 'static>(path: P) -> T {
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("Failed to read file at path: {}", path.as_ref().display()));

    toml::from_str(&content).unwrap_or_else(|err| {
        panic!(
            "Failed to deserialize TOML file {}: {err}",
            path.as_ref().display()
        )
    })
}

/// Returns the value for a command-line key from `--key value` or `--key=value`.
pub fn get_arg_value(args: &[String], key: &str) -> Option<String> {
    for i in 0..args.len() {
        if args[i] == key {
            return args.get(i + 1).cloned();
        }
        if let Some(v) = args[i].strip_prefix(&format!("{key}=")) {
            return Some(v.to_string());
        }
    }
    None
}

/// Returns the default configuration directory for examples and binaries.
///
/// The current working directory's `config` directory is preferred. If it is
/// absent, the crate-local `config` directory is used when present. Otherwise,
/// this returns the current working directory's `config` path even if it does
/// not exist.
pub fn default_cfg_dir() -> PathBuf {
    // Prefer a local "./config" (works for standalone bins), otherwise fall back to the
    // crate's config directory (works when running from workspace root via cargo).
    let cwd_config = PathBuf::from("config");
    if cwd_config.is_dir() {
        return cwd_config;
    }

    let manifest_config = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("config");
    if manifest_config.is_dir() {
        return manifest_config;
    }

    cwd_config
}

#[cfg(feature = "model")]
/// Model task families supported by model configuration.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ModelTypeOptions {
    /// Autoregressive language-model style generation.
    AutoRegressive,
    /// Sequence embedding model usage.
    SequenceEmbedding,
}

#[cfg(feature = "train")]
/// Optimizer choices supported by training configuration.
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerOptions {
    /// AdamW optimizer.
    #[default]
    AdamW,
}

#[cfg(feature = "train")]
/// Dataset file format choices supported by training configuration.
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetFormatOptions {
    /// Current RWKV dataset format.
    #[default]
    Rwkv,
    /// Legacy RWKV dataset format.
    RwkvLegacy,
}

#[cfg(feature = "train")]
/// Token unit storage types used by memory-mapped training datasets.
#[derive(
    Clone, Copy, Debug, Default, serde::Deserialize, serde::Serialize, Eq, PartialEq, Hash,
)]
#[repr(u8)]
pub enum TokenUnitDType {
    /// Unsigned 8-bit integer token unit.
    U8 = 0,
    /// Unsigned 16-bit integer token unit.
    #[default]
    U16 = 1,
    /// 32-bit floating-point token unit.
    F32 = 2,
}

#[cfg(feature = "train")]
impl TokenUnitDType {
    /// Returns whether this dtype stores discrete token ids.
    pub fn is_discrete(&self) -> bool {
        matches!(self, TokenUnitDType::U8 | TokenUnitDType::U16)
    }

    /// Converts the serialized dtype code into a token unit dtype.
    ///
    /// # Panics
    ///
    /// Panics when `code` is not one of the supported dtype codes.
    pub fn get_dtype(code: u8) -> Self {
        match code {
            0 => TokenUnitDType::U8,
            1 => TokenUnitDType::U16,
            2 => TokenUnitDType::F32,
            _ => panic!("Unsupported DTYPE code: {}", code),
        }
    }

    /// Returns the byte size of one stored token unit for this dtype.
    pub fn get_token_unit_size(&self) -> usize {
        match self {
            TokenUnitDType::U8 => std::mem::size_of::<u8>(),
            TokenUnitDType::U16 => std::mem::size_of::<u16>(),
            TokenUnitDType::F32 => std::mem::size_of::<f32>(),
        }
    }
}
