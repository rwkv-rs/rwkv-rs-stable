use burn::tensor::{
    backend::Backend,
    ops::{FloatTensor, IntTensor},
};

/// Primitive tensor inputs passed to the fused LM-head L2Wrap cross-entropy kernel.
#[derive(Clone)]
pub struct LmHeadL2WrapCePrimitiveInputs<B: Backend> {
    /// Logits shaped `[batch_size, context_len, vocab_size]`.
    pub logits: FloatTensor<B>,
    /// Target token ids shaped `[batch_size, context_len]`.
    pub targets: IntTensor<B>,
}
