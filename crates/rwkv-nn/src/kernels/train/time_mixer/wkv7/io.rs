use burn::tensor::{Tensor, ops::FloatTensor};

use crate::kernels::{
    check::{
        KernelInputsError,
        check_axes_equal,
        check_axis_non_empty,
        check_same_device,
        check_same_dtype,
        check_shape,
        get_tensor_info,
    },
    train::time_mixer::wkv7::Wkv7Backend as Backend,
};

/// Public tensor inputs for RWKV7 pretrain WKV.
#[derive(Debug, Clone)]
pub struct Wkv7PretrainForwardInputs<B: Backend> {
    /// Receptance shaped `[batch_size, context_len, num_heads, head_size]`.
    pub receptance: Tensor<B, 4>,
    /// Weight-decay precursor shaped `[batch_size, context_len, num_heads, head_size]`.
    pub weight_decay: Tensor<B, 4>,
    /// Replacement key shaped `[batch_size, context_len, num_heads, head_size]`.
    pub replacement_key: Tensor<B, 4>,
    /// Value shaped `[batch_size, context_len, num_heads, head_size]`.
    pub value: Tensor<B, 4>,
    /// Normalized removal key shaped `[batch_size, context_len, num_heads, head_size]`.
    pub removal_key_normalized: Tensor<B, 4>,
    /// Replacement shaped `[batch_size, context_len, num_heads, head_size]`.
    pub replacement: Tensor<B, 4>,
    /// Recurrence chunk length. The first implementation supports `16`.
    pub chunk_len: usize,
}

impl<B> Wkv7PretrainForwardInputs<B>
where
    B: Backend,
{
    pub(crate) fn to_primitive(&self) -> Wkv7PretrainForwardPrimitiveInputs<B> {
        Wkv7PretrainForwardPrimitiveInputs {
            receptance: self.receptance.clone().into_primitive().tensor(),
            weight_decay: self.weight_decay.clone().into_primitive().tensor(),
            replacement_key: self.replacement_key.clone().into_primitive().tensor(),
            value: self.value.clone().into_primitive().tensor(),
            removal_key_normalized: self
                .removal_key_normalized
                .clone()
                .into_primitive()
                .tensor(),
            replacement: self.replacement.clone().into_primitive().tensor(),
            chunk_len: self.chunk_len,
        }
    }

    pub(crate) fn check(&self) -> Result<(), KernelInputsError<B>> {
        check_wkv7_sequence_inputs(
            &self.receptance,
            &self.weight_decay,
            &self.replacement_key,
            &self.value,
            &self.removal_key_normalized,
            &self.replacement,
            self.chunk_len,
        )
    }
}

/// Primitive tensor inputs for RWKV7 pretrain WKV.
#[derive(Debug, Clone)]
pub struct Wkv7PretrainForwardPrimitiveInputs<B: Backend> {
    /// Primitive receptance.
    pub receptance: FloatTensor<B>,
    /// Primitive weight-decay precursor.
    pub weight_decay: FloatTensor<B>,
    /// Primitive replacement key.
    pub replacement_key: FloatTensor<B>,
    /// Primitive value.
    pub value: FloatTensor<B>,
    /// Primitive normalized removal key.
    pub removal_key_normalized: FloatTensor<B>,
    /// Primitive replacement.
    pub replacement: FloatTensor<B>,
    /// Recurrence chunk length.
    pub chunk_len: usize,
}

/// Public tensor inputs for RWKV7 StateTuning WKV.
#[derive(Debug, Clone)]
pub struct Wkv7StatetuneForwardInputs<B: Backend> {
    /// Initial state shaped `[batch_size, num_heads, head_size, head_size]`.
    pub initial_state: Tensor<B, 4>,
    /// Sequence inputs shared with the pretrain WKV.
    pub sequence: Wkv7PretrainForwardInputs<B>,
}

impl<B> Wkv7StatetuneForwardInputs<B>
where
    B: Backend,
{
    pub(crate) fn to_primitive(&self) -> Wkv7StatetuneForwardPrimitiveInputs<B> {
        Wkv7StatetuneForwardPrimitiveInputs {
            initial_state: self.initial_state.clone().into_primitive().tensor(),
            sequence: self.sequence.to_primitive(),
        }
    }

    pub(crate) fn check(&self) -> Result<(), KernelInputsError<B>> {
        self.sequence.check()?;
        check_initial_state(&self.initial_state, &self.sequence.receptance)
    }
}

/// Primitive tensor inputs for RWKV7 StateTuning WKV.
#[derive(Debug, Clone)]
pub struct Wkv7StatetuneForwardPrimitiveInputs<B: Backend> {
    /// Primitive initial state.
    pub initial_state: FloatTensor<B>,
    /// Primitive sequence inputs.
    pub sequence: Wkv7PretrainForwardPrimitiveInputs<B>,
}

/// Public tensor inputs for RWKV7 state-passing WKV.
#[derive(Debug, Clone)]
pub struct Wkv7StatepassForwardInputs<B: Backend> {
    /// Initial state shaped `[batch_size, num_heads, head_size, head_size]`.
    pub initial_state: Tensor<B, 4>,
    /// Sequence inputs shared with the pretrain WKV.
    pub sequence: Wkv7PretrainForwardInputs<B>,
}

impl<B> Wkv7StatepassForwardInputs<B>
where
    B: Backend,
{
    pub(crate) fn to_primitive(&self) -> Wkv7StatepassForwardPrimitiveInputs<B> {
        Wkv7StatepassForwardPrimitiveInputs {
            initial_state: self.initial_state.clone().into_primitive().tensor(),
            sequence: self.sequence.to_primitive(),
        }
    }

    pub(crate) fn check(&self) -> Result<(), KernelInputsError<B>> {
        self.sequence.check()?;
        check_initial_state(&self.initial_state, &self.sequence.receptance)
    }
}

/// Primitive tensor inputs for RWKV7 state-passing WKV.
#[derive(Debug, Clone)]
pub struct Wkv7StatepassForwardPrimitiveInputs<B: Backend> {
    /// Primitive initial state.
    pub initial_state: FloatTensor<B>,
    /// Primitive sequence inputs.
    pub sequence: Wkv7PretrainForwardPrimitiveInputs<B>,
}

/// Public output for RWKV7 state-passing WKV.
#[derive(Debug, Clone)]
pub struct Wkv7StatepassForwardOutput<B: Backend> {
    /// WKV output shaped `[batch_size, context_len, num_heads, head_size]`.
    pub output: Tensor<B, 4>,
    /// Final state shaped `[batch_size, num_heads, head_size, head_size]`.
    pub next_state: Tensor<B, 4>,
}

/// Primitive output for RWKV7 state-passing WKV.
#[derive(Debug, Clone)]
pub struct Wkv7StatepassForwardPrimitiveOutput<B: Backend> {
    /// Primitive WKV output.
    pub output: FloatTensor<B>,
    /// Primitive final state.
    pub next_state: FloatTensor<B>,
}

fn check_wkv7_sequence_inputs<B>(
    receptance: &Tensor<B, 4>,
    weight_decay: &Tensor<B, 4>,
    replacement_key: &Tensor<B, 4>,
    value: &Tensor<B, 4>,
    removal_key_normalized: &Tensor<B, 4>,
    replacement: &Tensor<B, 4>,
    chunk_len: usize,
) -> Result<(), KernelInputsError<B>>
where
    B: Backend,
{
    assert_eq!(chunk_len, 16, "wkv7 currently supports chunk_len=16");

    let expected_shape = receptance.shape();
    let receptance = get_tensor_info("receptance", receptance);
    let weight_decay = get_tensor_info("weight_decay", weight_decay);
    let replacement_key = get_tensor_info("replacement_key", replacement_key);
    let value = get_tensor_info("value", value);
    let removal_key_normalized = get_tensor_info("removal_key_normalized", removal_key_normalized);
    let replacement = get_tensor_info("replacement", replacement);
    let context_len = receptance.dim(1);

    assert!(
        context_len.is_multiple_of(chunk_len),
        "context_len must be a multiple of chunk_len"
    );
    check_axis_non_empty(receptance.axis(1))?;
    check_axis_non_empty(receptance.axis(3))?;
    check_axes_equal(&[
        receptance.axis(0),
        weight_decay.axis(0),
        replacement_key.axis(0),
        value.axis(0),
        removal_key_normalized.axis(0),
        replacement.axis(0),
    ])?;
    check_shape(&weight_decay, expected_shape.clone())?;
    check_shape(&replacement_key, expected_shape.clone())?;
    check_shape(&value, expected_shape.clone())?;
    check_shape(&removal_key_normalized, expected_shape.clone())?;
    check_shape(&replacement, expected_shape)?;
    check_same_dtype(&[
        &receptance,
        &weight_decay,
        &replacement_key,
        &value,
        &removal_key_normalized,
        &replacement,
    ])?;
    check_same_device(&[
        &receptance,
        &weight_decay,
        &replacement_key,
        &value,
        &removal_key_normalized,
        &replacement,
    ])?;

    Ok(())
}

fn check_initial_state<B>(
    initial_state: &Tensor<B, 4>,
    receptance: &Tensor<B, 4>,
) -> Result<(), KernelInputsError<B>>
where
    B: Backend,
{
    let initial_state = get_tensor_info("initial_state", initial_state);
    let receptance = get_tensor_info("receptance", receptance);
    let batch_size = receptance.dim(0);
    let num_heads = receptance.dim(2);
    let head_size = receptance.dim(3);

    check_shape(
        &initial_state,
        [batch_size, num_heads, head_size, head_size],
    )?;
    check_same_dtype(&[&initial_state, &receptance])?;
    check_same_device(&[&initial_state, &receptance])?;

    Ok(())
}
