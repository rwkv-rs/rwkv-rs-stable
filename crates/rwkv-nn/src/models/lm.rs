use burn::{
    config::Config,
    module::Module,
    nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    prelude::{Backend, Int, Tensor},
};

use crate::{
    cells::causal::{MultiCausalCells, MultiCausalCellsConfig, MultiCausalCellsIO},
    functions::init_weights::{orthogonal_init, uniform_init},
    kernels::train::{TrainBackend, layer_norm::layer_norm, lm_head_l2wrap_ce::lm_head_l2wrap_ce},
    modules::time_mixer::param_state::{StateModule, StateModuleConfig},
};

/// Configuration for an RWKV language model.
#[derive(Config, Debug)]
pub struct RwkvLMConfig {
    num_cells: usize,
    vocab_size: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,
}

impl RwkvLMConfig {
    /// Initializes the model on `device`.
    pub fn init<B: Backend>(&self, device: &B::Device) -> RwkvLM<B> {
        RwkvLM {
            embed: EmbeddingConfig::new(self.vocab_size, self.embedded_dim).init(device),
            layer_norm_for_first_cell: LayerNormConfig::new(self.embedded_dim).init(device),
            cells: MultiCausalCellsConfig::new(
                self.num_cells,
                self.embedded_dim,
                self.num_heads,
                self.head_size,
            )
            .init(device),
            layer_norm_for_unembed: LayerNormConfig::new(self.embedded_dim).init(device),
            unembed: LinearConfig::new(self.embedded_dim, self.vocab_size)
                .with_bias(false)
                .init(device),
            state: StateModuleConfig::new(self.num_cells, self.num_heads, self.head_size)
                .init(device),

            num_cells: self.num_cells,
            vocab_size: self.vocab_size,
            embedded_dim: self.embedded_dim,
            num_heads: self.num_heads,
            head_size: self.head_size,
        }
    }
}

/// RWKV language model used by training and checkpoint conversion paths.
#[derive(Module, Debug)]
pub struct RwkvLM<B: Backend> {
    /// Token embedding table.
    pub embed: Embedding<B>,
    /// Layer norm applied before the first causal cell.
    pub layer_norm_for_first_cell: LayerNorm<B>,
    /// Ordered stack of recurrent causal cells.
    pub cells: MultiCausalCells<B>,
    /// Layer norm applied before the unembedding projection.
    pub layer_norm_for_unembed: LayerNorm<B>,
    /// Output projection from hidden states to vocabulary logits.
    pub unembed: Linear<B>,
    /// Trainable initial recurrent state.
    pub state: StateModule<B>,

    num_cells: usize,
    vocab_size: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,
}

impl<B: Backend> RwkvLM<B> {
    /// Initializes trainable weights in place.
    pub fn init_weights(&mut self, device: &B::Device) {
        uniform_init(&mut self.embed.weight, -1e-4, 1e-4);

        if self.vocab_size > self.embedded_dim {
            orthogonal_init(
                &mut self.unembed.weight,
                Some(0.5 * (self.vocab_size as f32 / self.embedded_dim as f32).sqrt()),
            );
        } else {
            orthogonal_init(&mut self.unembed.weight, Some(0.5));
        }

        self.cells.init_weights(device);
    }

    /// Runs a training forward pass and returns L2Wrap cross-entropy loss.
    pub fn forward(
        &self,
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
        state: Option<Vec<Tensor<B, 4>>>,
        embedded_token_shift_for_channel_mix: Option<Vec<Tensor<B, 2>>>,
    ) -> RwkvLMForwardOutput<B>
    where
        B: TrainBackend,
    {
        if let Some(state) = state.as_ref() {
            debug_assert_eq!(state.len(), self.num_cells);
            debug_assert!(state.iter().all(|s| {
                let [_batch_size, num_heads, head_size_left, head_size_right] = s.dims();
                num_heads == self.num_heads
                    && head_size_left == self.head_size
                    && head_size_right == self.head_size
            }));
        }
        if let Some(token_shift) = embedded_token_shift_for_channel_mix.as_ref() {
            debug_assert_eq!(token_shift.len(), self.num_cells);
            debug_assert!(token_shift.iter().all(|t| t.dims()[1] == self.embedded_dim));
        }

        let RwkvLMLogitsOutput {
            logits,
            state,
            embedded_token_shift_for_channel_mix,
        } = self.forward_logits(inputs, state, embedded_token_shift_for_channel_mix);
        let loss = lm_head_l2wrap_ce(logits, targets);

        RwkvLMForwardOutput {
            loss,
            state,
            embedded_token_shift_for_channel_mix,
        }
    }

    /// Runs the model body and returns raw vocabulary logits.
    pub fn forward_logits(
        &self,
        inputs: Tensor<B, 2, Int>,
        state: Option<Vec<Tensor<B, 4>>>,
        embedded_token_shift_for_channel_mix: Option<Vec<Tensor<B, 2>>>,
    ) -> RwkvLMLogitsOutput<B>
    where
        B: TrainBackend,
    {
        let embedded_context = self.embed.forward(inputs);
        let embedded_context_normalized = layer_norm(
            embedded_context,
            self.layer_norm_for_first_cell.gamma.val(),
            self.layer_norm_for_first_cell
                .beta
                .as_ref()
                .expect("rwkv lm layer norm requires affine beta")
                .val(),
            1e-5,
        );
        let multi_causal_cells_output = self.cells.forward(MultiCausalCellsIO {
            embedded_context: embedded_context_normalized,
            state,
            embedded_token_shift_for_channel_mix,
        });
        let embedded_context_normalized = layer_norm(
            multi_causal_cells_output.embedded_context,
            self.layer_norm_for_unembed.gamma.val(),
            self.layer_norm_for_unembed
                .beta
                .as_ref()
                .expect("rwkv lm layer norm requires affine beta")
                .val(),
            1e-5,
        );
        let logits = self.unembed.forward(embedded_context_normalized);

        RwkvLMLogitsOutput {
            logits,
            state: multi_causal_cells_output.state,
            embedded_token_shift_for_channel_mix: multi_causal_cells_output
                .embedded_token_shift_for_channel_mix,
        }
    }
}

/// Output tensors from [`RwkvLM::forward`].
pub struct RwkvLMForwardOutput<B: Backend> {
    /// Scalar L2Wrap cross-entropy loss.
    pub loss: Tensor<B, 1>,
    /// Optional next recurrent state for each cell.
    pub state: Option<Vec<Tensor<B, 4>>>,
    /// Optional next channel-mix token-shift embedding for each cell.
    pub embedded_token_shift_for_channel_mix: Option<Vec<Tensor<B, 2>>>,
}

/// Output tensors from [`RwkvLM::forward_logits`].
pub struct RwkvLMLogitsOutput<B: Backend> {
    /// Vocabulary logits with shape `[batch_size, context_length, vocab_size]`.
    pub logits: Tensor<B, 3>,
    /// Optional next recurrent state for each cell.
    pub state: Option<Vec<Tensor<B, 4>>>,
    /// Optional next channel-mix token-shift embedding for each cell.
    pub embedded_token_shift_for_channel_mix: Option<Vec<Tensor<B, 2>>>,
}
