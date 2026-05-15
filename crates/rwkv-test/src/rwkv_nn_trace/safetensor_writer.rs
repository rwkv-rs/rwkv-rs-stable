use std::{borrow::Cow, collections::HashMap, fs, path::Path};

use anyhow::{Context, Result, bail};
use burn::tensor::{DType, TensorData};
use safetensors::tensor::{Dtype as SafeDtype, View, serialize_to_file};

pub(super) fn write_safetensor(path: &Path, data: TensorData) -> Result<()> {
    let parent = path.parent().expect("trace file has a parent directory");
    fs::create_dir_all(parent).with_context(|| format!("failed to create {}", parent.display()))?;
    let name = path
        .file_stem()
        .and_then(|name| name.to_str())
        .expect("trace file stem is valid UTF-8")
        .to_owned();
    let tensor = TraceTensor::from_data(data)?;
    serialize_to_file(HashMap::from([(name, tensor)]), None, path)
        .with_context(|| format!("failed to write {}", path.display()))
}

struct TraceTensor {
    dtype: SafeDtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl TraceTensor {
    fn from_data(data: TensorData) -> Result<Self> {
        let dtype = safe_dtype(data.dtype)?;
        Ok(Self {
            dtype,
            shape: data.shape.to_vec(),
            data: data.as_bytes().to_vec(),
        })
    }
}

impl View for TraceTensor {
    fn dtype(&self) -> SafeDtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.data)
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

fn safe_dtype(dtype: DType) -> Result<SafeDtype> {
    match dtype {
        DType::F64 => Ok(SafeDtype::F64),
        DType::F32 | DType::Flex32 => Ok(SafeDtype::F32),
        DType::F16 => Ok(SafeDtype::F16),
        DType::BF16 => Ok(SafeDtype::BF16),
        DType::I64 => Ok(SafeDtype::I64),
        DType::I32 => Ok(SafeDtype::I32),
        DType::I16 => Ok(SafeDtype::I16),
        DType::I8 => Ok(SafeDtype::I8),
        DType::U64 => Ok(SafeDtype::U64),
        DType::U32 => Ok(SafeDtype::U32),
        DType::U16 => Ok(SafeDtype::U16),
        DType::U8 => Ok(SafeDtype::U8),
        DType::Bool(_) => Ok(SafeDtype::BOOL),
        DType::QFloat(_) => bail!("quantized trace tensors are not supported"),
    }
}
