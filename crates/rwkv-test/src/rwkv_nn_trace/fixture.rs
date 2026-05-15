use std::path::Path;

use anyhow::{Context, Result, bail};
use burn::{
    prelude::{Int, Tensor},
    tensor::TensorData,
};

use crate::read_safetensor_values;
use super::{BATCH_SIZE, CONTEXT_LEN, TraceBackend, TraceDevice};

pub(super) fn trace_tokens(
    root: &Path,
    device: &TraceDevice,
) -> Result<Tensor<TraceBackend, 2, Int>> {
    let (dtype, shape, values) =
        read_safetensor_values(root.join("embedding/token_ids.safetensors"))
            .context("failed to read baseline token ids")?;
    if dtype.to_string() != "I64" {
        bail!("embedding/token_ids.safetensors must be I64, got {dtype}");
    }
    if shape != [BATCH_SIZE, CONTEXT_LEN] {
        bail!(
            "embedding/token_ids.safetensors shape must be [{BATCH_SIZE},{CONTEXT_LEN}], got {:?}",
            shape
        );
    }

    let tokens = values
        .into_iter()
        .map(|value| value as i32)
        .collect::<Vec<_>>();
    Ok(Tensor::from_ints(
        TensorData::new(tokens, [BATCH_SIZE, CONTEXT_LEN]),
        device,
    ))
}
