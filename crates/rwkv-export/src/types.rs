use burn::tensor::DType;
use safetensors::tensor::SafeTensorError;
use thiserror::Error;

/// Options used when writing RWKV LM `.safetensors` files.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ConvertOptions {
    /// Replace the output file when it already exists.
    pub overwrite: bool,
}

/// Shape information inferred from a RWKV LM `.safetensors` file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RwkvLmStInfo {
    /// Number of tensors in the file.
    pub num_tensors: usize,
    /// Number of RWKV blocks.
    pub num_cells: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Embedding dimension.
    pub embedded_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Per-head dimension.
    pub head_size: usize,
}

/// Errors reported by RWKV export operations.
#[derive(Debug, Error)]
pub enum ExportError {
    /// Input/output error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// PyTorch checkpoint loading failed.
    #[error("PyTorch checkpoint error: {0}")]
    Pytorch(String),
    /// Safetensors parsing or writing failed.
    #[error("safetensors error: {0}")]
    Safetensors(#[from] SafeTensorError),
    /// Burn safetensors store failed.
    #[error("Burn safetensors store error: {0}")]
    BurnSafetensors(String),
    /// Tensor materialization failed.
    #[error("tensor data error: {0}")]
    TensorData(String),
    /// The input is outside the supported RWKV7 G1 scope.
    #[error("unsupported RWKV checkpoint layout: {0}")]
    UnsupportedLayout(String),
    /// A required tensor is absent or has an invalid shape.
    #[error("invalid RWKV7 G1 tensor set: {0}")]
    InvalidTensorSet(String),
    /// A tensor has a dtype this exporter does not handle.
    #[error("unsupported tensor dtype for {name}: {dtype:?}")]
    UnsupportedDType {
        /// Tensor name.
        name: String,
        /// Tensor dtype.
        dtype: DType,
    },
}
