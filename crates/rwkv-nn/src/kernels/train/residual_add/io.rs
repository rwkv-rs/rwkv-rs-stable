use burn::tensor::ops::FloatTensor;

/// Primitive tensors consumed by the fused residual-add kernel.
pub struct ResidualAddPrimitiveInputs<B: burn::tensor::backend::Backend> {
    /// Left-hand activation tensor.
    pub lhs: FloatTensor<B>,
    /// Right-hand residual tensor.
    pub rhs: FloatTensor<B>,
}
