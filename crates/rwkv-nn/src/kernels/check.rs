use burn::tensor::{backend::Backend, BasicOps, DType, Shape, Tensor};
use thiserror::Error;

#[derive(Clone, Debug)]
pub(crate) struct TensorInfo<B: Backend> {
    name: &'static str,
    shape: Shape,
    dtype: DType,
    device: B::Device,
}

impl<B: Backend> TensorInfo<B> {
    pub(crate) fn axis(&self, axis: usize) -> AxisInfo<'_, B> {
        AxisInfo { tensor: self, axis }
    }

    pub(crate) fn dim(&self, axis: usize) -> usize {
        self.shape[axis]
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct AxisInfo<'a, B: Backend> {
    tensor: &'a TensorInfo<B>,
    axis: usize,
}

impl<'a, B: Backend> AxisInfo<'a, B> {
    fn dim(&self) -> usize {
        self.tensor.dim(self.axis)
    }
}

#[derive(Debug, Error)]
pub(crate) enum KernelInputsError<B: Backend> {
    #[error(
        "kernel input axis mismatch: {reference_name}[{reference_axis}]={reference_dim} != {tensor_name}[{tensor_axis}]={tensor_dim}"
    )]
    AxisMismatch {
        reference_name: &'static str,
        reference_axis: usize,
        reference_dim: usize,
        tensor_name: &'static str,
        tensor_axis: usize,
        tensor_dim: usize,
    },
    #[error("kernel input shape mismatch for {tensor}: expected={expected:?}, actual={actual:?}")]
    ShapeMismatch {
        tensor: &'static str,
        expected: Shape,
        actual: Shape,
    },
    #[error(
        "kernel input dtype mismatch: {reference_name}={reference:?}, {tensor_name}={tensor:?}"
    )]
    DTypeMismatch {
        reference_name: &'static str,
        reference: DType,
        tensor_name: &'static str,
        tensor: DType,
    },
    #[error(
        "kernel input device mismatch: {reference_name}={reference:?}, {tensor_name}={tensor:?}"
    )]
    DeviceMismatch {
        reference_name: &'static str,
        reference: B::Device,
        tensor_name: &'static str,
        tensor: B::Device,
    },
    #[error("kernel input axis must be non-empty: {tensor}[{axis}]")]
    EmptyAxis { tensor: &'static str, axis: usize },
}

pub(crate) fn get_tensor_info<B, const D: usize, K>(
    name: &'static str,
    tensor: &Tensor<B, D, K>,
) -> TensorInfo<B>
where
    B: Backend,
    K: BasicOps<B>,
{
    TensorInfo {
        name,
        shape: tensor.shape(),
        dtype: tensor.dtype(),
        device: tensor.device(),
    }
}

pub(crate) fn check_axes_equal<B>(axes: &[AxisInfo<'_, B>]) -> Result<(), KernelInputsError<B>>
where
    B: Backend,
{
    let Some((reference, rest)) = axes.split_first() else {
        return Ok(());
    };

    let reference_dim = reference.dim();
    for axis in rest {
        let tensor_dim = axis.dim();
        if tensor_dim != reference_dim {
            return Err(KernelInputsError::AxisMismatch {
                reference_name: reference.tensor.name,
                reference_axis: reference.axis,
                reference_dim,
                tensor_name: axis.tensor.name,
                tensor_axis: axis.axis,
                tensor_dim,
            });
        }
    }

    Ok(())
}

pub(crate) fn check_shape<B, S>(
    tensor: &TensorInfo<B>,
    expected: S,
) -> Result<(), KernelInputsError<B>>
where
    B: Backend,
    S: Into<Shape>,
{
    let expected = expected.into();
    let actual = tensor.shape.clone();

    if actual != expected {
        return Err(KernelInputsError::ShapeMismatch {
            tensor: tensor.name,
            expected,
            actual,
        });
    }

    Ok(())
}

pub(crate) fn check_axis_non_empty<B>(axis: AxisInfo<'_, B>) -> Result<(), KernelInputsError<B>>
where
    B: Backend,
{
    if axis.dim() == 0 {
        return Err(KernelInputsError::EmptyAxis {
            tensor: axis.tensor.name,
            axis: axis.axis,
        });
    }

    Ok(())
}

#[allow(dead_code)]
pub(crate) fn check_same_shape<B>(inputs: &[&TensorInfo<B>]) -> Result<(), KernelInputsError<B>>
where
    B: Backend,
{
    let Some((reference, rest)) = inputs.split_first() else {
        return Ok(());
    };

    for input in rest {
        if input.shape != reference.shape {
            return Err(KernelInputsError::ShapeMismatch {
                tensor: input.name,
                expected: reference.shape.clone(),
                actual: input.shape.clone(),
            });
        }
    }

    Ok(())
}

pub(crate) fn check_same_dtype<B>(inputs: &[&TensorInfo<B>]) -> Result<(), KernelInputsError<B>>
where
    B: Backend,
{
    let Some((reference, rest)) = inputs.split_first() else {
        return Ok(());
    };

    for input in rest {
        if input.dtype != reference.dtype {
            return Err(KernelInputsError::DTypeMismatch {
                reference_name: reference.name,
                reference: reference.dtype,
                tensor_name: input.name,
                tensor: input.dtype,
            });
        }
    }

    Ok(())
}

pub(crate) fn check_same_device<B>(inputs: &[&TensorInfo<B>]) -> Result<(), KernelInputsError<B>>
where
    B: Backend,
{
    let Some((reference, rest)) = inputs.split_first() else {
        return Ok(());
    };

    for input in rest {
        if input.device != reference.device {
            return Err(KernelInputsError::DeviceMismatch {
                reference_name: reference.name,
                reference: reference.device.clone(),
                tensor_name: input.name,
                tensor: input.device.clone(),
            });
        }
    }

    Ok(())
}
