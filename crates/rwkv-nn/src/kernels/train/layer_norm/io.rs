use burn::tensor::ops::FloatTensor;

/// Primitive tensors consumed by the fused layer-normalization kernel.
pub struct LayerNormPrimitiveInputs<B: burn::tensor::backend::Backend> {
    /// Input tensor shaped `[batch_size, context_len, embedded_dim]`.
    pub input: FloatTensor<B>,
    /// Affine scale vector shaped `[embedded_dim]`.
    pub gamma: FloatTensor<B>,
    /// Affine bias vector shaped `[embedded_dim]`.
    pub beta: FloatTensor<B>,
    /// Numerical-stability epsilon added to the variance.
    pub epsilon: f64,
}
