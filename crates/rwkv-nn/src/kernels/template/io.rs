use burn::tensor::{ops::FloatTensor, Tensor};

use crate::kernels::{
    check::{
        check_axes_equal,
        check_same_device,
        check_same_dtype,
        check_shape,
        get_tensor_info,
        KernelInputsError,
    },
    template::TemplateBackend as Backend,
};

#[derive(Debug, Clone)]
pub struct MatmulAddReluInputs<B: Backend> {
    pub lhs: Tensor<B, 3>,
    pub rhs: Tensor<B, 3>,
    pub bias: Tensor<B, 3>,
}

impl<B> MatmulAddReluInputs<B>
where
    B: Backend,
{
    pub(crate) fn to_primitive(&self) -> MatmulAddReluPrimitiveInputs<B> {
        MatmulAddReluPrimitiveInputs {
            lhs: self.lhs.clone().into_primitive().tensor(),
            rhs: self.rhs.clone().into_primitive().tensor(),
            bias: self.bias.clone().into_primitive().tensor(),
        }
    }
}

impl<B> MatmulAddReluInputs<B>
where
    B: Backend,
{
    pub(crate) fn check(&self) -> Result<(), KernelInputsError<B>> {
        let lhs = get_tensor_info("lhs", &self.lhs);
        let rhs = get_tensor_info("rhs", &self.rhs);
        let bias = get_tensor_info("bias", &self.bias);

        check_axes_equal(&[lhs.axis(2), rhs.axis(1)])?;

        let lhs_batch = lhs.dim(0);
        let rhs_batch = rhs.dim(0);
        if lhs_batch != rhs_batch && lhs_batch != 1 && rhs_batch != 1 {
            return Err(KernelInputsError::AxisMismatch {
                reference_name: "lhs",
                reference_axis: 0,
                reference_dim: lhs_batch,
                tensor_name: "rhs",
                tensor_axis: 0,
                tensor_dim: rhs_batch,
            });
        }

        check_shape(&bias, [lhs.dim(0).max(rhs.dim(0)), lhs.dim(1), rhs.dim(2)])?;
        check_same_dtype(&[&lhs, &rhs, &bias])?;
        check_same_device(&[&lhs, &rhs, &bias])?;

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct MatmulAddReluPrimitiveInputs<B: Backend> {
    pub lhs: FloatTensor<B>,
    pub rhs: FloatTensor<B>,
    pub bias: FloatTensor<B>,
}
