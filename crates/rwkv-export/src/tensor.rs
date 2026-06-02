use std::{borrow::Cow, rc::Rc};

use burn::tensor::{DType, Shape, TensorData};
use burn_store::{ModuleAdapter, TensorSnapshot, TensorSnapshotError};
use safetensors::tensor::{Dtype as SafeDtype, View};

use crate::{
    ExportError,
    keys::{should_transpose_burn_path, should_transpose_st_key},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AdapterDirection {
    StToBurn,
    BurnToSt,
}

#[derive(Clone, Debug)]
pub(crate) struct RwkvLmStAdapter {
    direction: AdapterDirection,
    load_dtype: Option<DType>,
}

impl RwkvLmStAdapter {
    pub(crate) fn new(direction: AdapterDirection) -> Self {
        Self {
            direction,
            load_dtype: None,
        }
    }

    pub(crate) fn with_load_dtype(mut self, dtype: DType) -> Self {
        self.load_dtype = Some(dtype);
        self
    }
}

impl ModuleAdapter for RwkvLmStAdapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        let mut adapted = snapshot.clone();
        if should_transpose_adapter_snapshot(snapshot) {
            adapted = transpose_snapshot(&adapted);
        }

        match self.direction {
            AdapterDirection::StToBurn => {
                if let Some(dtype) = self.load_dtype {
                    adapted = cast_snapshot_dtype(&adapted, dtype);
                }
            }
            AdapterDirection::BurnToSt => {
                adapted = cast_snapshot_dtype(&adapted, DType::F16);
            }
        }

        adapted
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

pub(crate) fn materialize_st_tensor(
    name: &str,
    snapshot: &TensorSnapshot,
) -> Result<OwnedTensor, ExportError> {
    let mut data = snapshot
        .to_data()
        .map_err(|error| ExportError::TensorData(error.to_string()))?;
    validate_float_dtype(name, data.dtype)?;

    if should_transpose_st_key(name) {
        data = transpose_last_two_dims(data).map_err(ExportError::TensorData)?;
    }
    data = data.convert_dtype(DType::F16);

    OwnedTensor::from_tensor_data(data)
}

pub(crate) fn validate_float_dtype(name: &str, dtype: DType) -> Result<(), ExportError> {
    if matches!(
        dtype,
        DType::F64 | DType::F32 | DType::Flex32 | DType::F16 | DType::BF16
    ) {
        Ok(())
    } else {
        Err(ExportError::UnsupportedDType {
            name: name.to_string(),
            dtype,
        })
    }
}

pub(crate) fn transpose_last_two_dims(data: TensorData) -> Result<TensorData, String> {
    let rank = data.shape.len();
    if rank < 2 {
        return Err(format!(
            "cannot transpose rank {} tensor with shape {:?}",
            rank, data.shape
        ));
    }

    let rows = data.shape[rank - 2];
    let cols = data.shape[rank - 1];
    let outer = data.shape.iter().take(rank - 2).product::<usize>();
    let elem_size = data.dtype.size();
    let bytes = data.as_bytes();
    let mut out = vec![0; bytes.len()];
    let matrix_bytes = rows * cols * elem_size;

    for outer_index in 0..outer {
        let base = outer_index * matrix_bytes;
        for row in 0..rows {
            for col in 0..cols {
                let src = base + (row * cols + col) * elem_size;
                let dst = base + (col * rows + row) * elem_size;
                out[dst..dst + elem_size].copy_from_slice(&bytes[src..src + elem_size]);
            }
        }
    }

    let mut shape = data.shape.to_vec();
    shape.swap(rank - 2, rank - 1);
    Ok(TensorData::from_bytes_vec(out, shape, data.dtype))
}

fn should_transpose_adapter_snapshot(snapshot: &TensorSnapshot) -> bool {
    let Some(path_stack) = snapshot.path_stack.as_ref() else {
        return false;
    };
    let path = path_stack.join(".");
    let module_type = snapshot.module_type();

    (module_type.as_deref() == Some("Struct:Linear")
        && path.ends_with(".weight")
        && snapshot.shape.len() == 2)
        || should_transpose_burn_path(&path)
}

fn transpose_snapshot(snapshot: &TensorSnapshot) -> TensorSnapshot {
    let mut shape = snapshot.shape.clone();
    let rank = shape.len();
    if rank >= 2 {
        let mut dims = shape.to_vec();
        dims.swap(rank - 2, rank - 1);
        shape = Shape::from(dims);
    }
    let data_fn = snapshot.clone_data_fn();
    let transpose_fn = Rc::new(move || {
        let data = data_fn()?;
        transpose_last_two_dims(data).map_err(TensorSnapshotError::DataError)
    });

    TensorSnapshot::from_closure(
        transpose_fn,
        snapshot.dtype,
        shape,
        snapshot.path_stack.clone().unwrap_or_default(),
        snapshot.container_stack.clone().unwrap_or_default(),
        snapshot.tensor_id.unwrap_or_default(),
    )
}

fn cast_snapshot_dtype(snapshot: &TensorSnapshot, dtype: DType) -> TensorSnapshot {
    if snapshot.dtype == dtype {
        return snapshot.clone();
    }

    let data_fn = snapshot.clone_data_fn();
    let cast_fn = Rc::new(move || {
        let data = data_fn()?;
        validate_float_dtype("snapshot", data.dtype)
            .map_err(|error| TensorSnapshotError::DataError(error.to_string()))?;
        Ok(data.convert_dtype(dtype))
    });

    TensorSnapshot::from_closure(
        cast_fn,
        dtype,
        snapshot.shape.clone(),
        snapshot.path_stack.clone().unwrap_or_default(),
        snapshot.container_stack.clone().unwrap_or_default(),
        snapshot.tensor_id.unwrap_or_default(),
    )
}

pub(crate) struct OwnedTensor {
    dtype: SafeDtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl OwnedTensor {
    fn from_tensor_data(data: TensorData) -> Result<Self, ExportError> {
        let dtype = burn_dtype_to_safe_dtype(data.dtype).ok_or(ExportError::UnsupportedDType {
            name: "tensor".to_string(),
            dtype: data.dtype,
        })?;

        Ok(Self {
            dtype,
            shape: data.shape.to_vec(),
            data: data.as_bytes().to_vec(),
        })
    }
}

impl View for OwnedTensor {
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

fn burn_dtype_to_safe_dtype(dtype: DType) -> Option<SafeDtype> {
    match dtype {
        DType::F64 => Some(SafeDtype::F64),
        DType::F32 | DType::Flex32 => Some(SafeDtype::F32),
        DType::F16 => Some(SafeDtype::F16),
        DType::BF16 => Some(SafeDtype::BF16),
        DType::I64 => Some(SafeDtype::I64),
        DType::I32 => Some(SafeDtype::I32),
        DType::I16 => Some(SafeDtype::I16),
        DType::I8 => Some(SafeDtype::I8),
        DType::U64 => Some(SafeDtype::U64),
        DType::U32 => Some(SafeDtype::U32),
        DType::U16 => Some(SafeDtype::U16),
        DType::U8 => Some(SafeDtype::U8),
        DType::Bool(_) => Some(SafeDtype::BOOL),
        DType::QFloat(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use burn::module::ParamId;
    use half::f16;

    use super::*;

    #[test]
    fn transpose_last_two_dims() {
        let data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let data = super::transpose_last_two_dims(data).unwrap();

        assert_eq!(data.shape.to_vec(), vec![3, 2]);
        assert_eq!(
            data.into_vec::<f32>().unwrap(),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }

    #[test]
    fn materialize_st_tensor_uses_fp16_and_st_transpose() {
        let data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let snapshot =
            TensorSnapshot::from_data(data, vec!["blocks".into()], vec![], ParamId::new());
        let tensor = super::materialize_st_tensor("blocks.0.att.w1", &snapshot).unwrap();

        assert_eq!(tensor.dtype, SafeDtype::F16);
        assert_eq!(tensor.shape, vec![3, 2]);
        let values = tensor
            .data
            .chunks_exact(2)
            .map(|bytes| f16::from_le_bytes([bytes[0], bytes[1]]).to_f32())
            .collect::<Vec<_>>();
        assert_eq!(values, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
