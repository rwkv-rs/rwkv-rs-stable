#![allow(dead_code)]

use burn::{prelude::Backend, tensor::Tensor};

pub const NUM_CELLS: usize = 2;
pub const VOCAB_SIZE: usize = 32;
pub const EMBEDDED_DIM: usize = 8;
pub const NUM_HEADS: usize = 2;
pub const HEAD_SIZE: usize = 4;
pub const CONTEXT_LEN: usize = 16;

pub fn zeros_context<B: Backend>(device: &B::Device) -> Tensor<B, 3> {
    Tensor::zeros([1, CONTEXT_LEN, EMBEDDED_DIM], device)
}

pub fn zeros_state<B: Backend>(device: &B::Device) -> Tensor<B, 4> {
    Tensor::zeros([1, NUM_HEADS, HEAD_SIZE, HEAD_SIZE], device)
}

pub fn zeros_token_shift<B: Backend>(device: &B::Device) -> Tensor<B, 2> {
    Tensor::zeros([1, EMBEDDED_DIM], device)
}
