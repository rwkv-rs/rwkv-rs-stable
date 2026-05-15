use std::{ffi::OsStr, fs, path::Path};

use anyhow::{Context, Result, bail};
use half::{bf16, f16};
use safetensors::tensor::{Dtype, SafeTensors, TensorView};

use crate::display::shape;

pub(crate) struct Tensor {
    pub(crate) dtype: Dtype,
    pub(crate) shape: Vec<usize>,
    pub(crate) bytes: Vec<u8>,
    pub(crate) values: Vec<f64>,
    pub(crate) exact: bool,
}

pub(crate) fn load(path: &Path) -> Result<std::result::Result<Tensor, String>> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let st = SafeTensors::deserialize(&bytes)
        .with_context(|| format!("failed to parse safetensors {}", path.display()))?;
    let tensors = st.tensors();
    if tensors.len() != 1 {
        return Ok(Err(format!(
            "expected exactly one tensor, found {}",
            tensors.len()
        )));
    }
    let (name, view) = tensors.into_iter().next().unwrap();
    let Some(stem) = path.file_stem().and_then(OsStr::to_str) else {
        return Ok(Err("file stem is not valid UTF-8".into()));
    };
    if name != stem {
        return Ok(Err(format!(
            "tensor name `{name}` does not match file stem `{stem}`"
        )));
    }
    Ok(decode(view))
}

/// Reads one single-tensor safetensors file and returns its dtype, shape, and decoded values.
pub fn read_safetensor_values(path: impl AsRef<Path>) -> Result<(Dtype, Vec<usize>, Vec<f64>)> {
    match load(path.as_ref())? {
        Ok(tensor) => Ok((tensor.dtype, tensor.shape, tensor.values)),
        Err(reason) => bail!("{reason}"),
    }
}

fn decode(view: TensorView<'_>) -> std::result::Result<Tensor, String> {
    let (raw, dtype, dims) = (view.data(), view.dtype(), view.shape().to_vec());
    let values = match dtype {
        Dtype::F64 => chunks(raw, 8, |b| f64::from_le_bytes(arr(b))),
        Dtype::F32 => chunks(raw, 4, |b| f32::from_le_bytes(arr(b)) as f64),
        Dtype::F16 => chunks(raw, 2, |b| {
            f16::from_bits(u16::from_le_bytes(arr(b))).to_f64()
        }),
        Dtype::BF16 => chunks(raw, 2, |b| {
            bf16::from_bits(u16::from_le_bytes(arr(b))).to_f64()
        }),
        Dtype::I64 => chunks(raw, 8, |b| i64::from_le_bytes(arr(b)) as f64),
        Dtype::I32 => chunks(raw, 4, |b| i32::from_le_bytes(arr(b)) as f64),
        Dtype::I16 => chunks(raw, 2, |b| i16::from_le_bytes(arr(b)) as f64),
        Dtype::I8 => raw.iter().map(|v| *v as i8 as f64).collect(),
        Dtype::U64 => chunks(raw, 8, |b| u64::from_le_bytes(arr(b)) as f64),
        Dtype::U32 => chunks(raw, 4, |b| u32::from_le_bytes(arr(b)) as f64),
        Dtype::U16 => chunks(raw, 2, |b| u16::from_le_bytes(arr(b)) as f64),
        Dtype::U8 => raw.iter().map(|v| f64::from(*v)).collect(),
        Dtype::BOOL => raw
            .iter()
            .map(|v| if *v == 0 { 0.0 } else { 1.0 })
            .collect(),
        Dtype::F4
        | Dtype::F6_E2M3
        | Dtype::F6_E3M2
        | Dtype::F8_E5M2
        | Dtype::F8_E4M3
        | Dtype::F8_E8M0
        | Dtype::C64 => return Err(format!("unsupported dtype {dtype}")),
        _ => return Err(format!("unsupported dtype {dtype}")),
    };

    let count = dims
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(*d))
        .ok_or_else(|| format!("shape {} overflows usize", shape(&dims)))?;
    if count != values.len() {
        return Err(format!(
            "shape {} expects {count} elements, decoded {}",
            shape(&dims),
            values.len()
        ));
    }

    // Integer and bool tensors usually carry ids or masks, so tolerance-based comparison is wrong.
    Ok(Tensor {
        dtype,
        shape: dims,
        bytes: raw.to_vec(),
        values,
        exact: matches!(
            dtype,
            Dtype::I64
                | Dtype::I32
                | Dtype::I16
                | Dtype::I8
                | Dtype::U64
                | Dtype::U32
                | Dtype::U16
                | Dtype::U8
                | Dtype::BOOL
        ),
    })
}

fn chunks(raw: &[u8], size: usize, f: impl Fn(&[u8]) -> f64) -> Vec<f64> {
    raw.chunks_exact(size).map(f).collect()
}

fn arr<const N: usize>(bytes: &[u8]) -> [u8; N] {
    bytes.try_into().unwrap()
}
