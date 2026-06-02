use rwkv::{
    custom::{
        Tensor,
        config::Config,
        module::Module,
        nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig},
        tensor::backend::{AutodiffBackend, Backend},
        train::{InferenceStep, TrainOutput, TrainStep},
    },
    nn::{
        cells::causal::{MultiCausalCells, MultiCausalCellsConfig, MultiCausalCellsIO},
        functions::init_weights::{orthogonal_init, uniform_init},
        kernels::train::{TrainBackend, lm_head_l2wrap_ce::lm_head_l2wrap_ce},
        modules::time_mixer::param_state::{StateModule, StateModuleConfig},
    },
    train::learner::next_token_prediction::NextTokenPredictionOutput,
};

use crate::data::batcher::AutoRegressiveBatch;

rwkv::custom_mode!();

#[derive(Config, Debug)]
pub struct AutoRegressiveModelConfig {
    num_cells: usize,
    vocab_size: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,
}

impl AutoRegressiveModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AutoRegressiveModel<B> {
        AutoRegressiveModel {
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

#[derive(Module, Debug)]
pub struct AutoRegressiveModel<B: Backend> {
    pub embed: Embedding<B>,
    pub layer_norm_for_first_cell: LayerNorm<B>,
    pub cells: MultiCausalCells<B>,
    pub layer_norm_for_unembed: LayerNorm<B>,
    pub unembed: Linear<B>,
    pub state: StateModule<B>,

    num_cells: usize,
    vocab_size: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,
}

impl<B: Backend> AutoRegressiveModel<B> {
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

    pub fn forward(
        &self,
        item: AutoRegressiveBatch<B>,
        state: Option<Vec<Tensor<B, 4>>>,
        embedded_token_shift_for_channel_mix: Option<Vec<Tensor<B, 2>>>,
    ) -> NextTokenPredictionOutput<B>
    where
        B: TrainBackend,
    {
        let device = &self.embed.devices()[0];
        let inputs = item.inputs.to_device(device);
        let targets = item.targets.to_device(device);

        let embedded_context = self.embed.forward(inputs);
        let embedded_context = self.layer_norm_for_first_cell.forward(embedded_context);
        let cells_output = self.cells.forward(MultiCausalCellsIO {
            embedded_context,
            state,
            embedded_token_shift_for_channel_mix,
        });
        let embedded_context = self
            .layer_norm_for_unembed
            .forward(cells_output.embedded_context);
        let logits = self.unembed.forward(embedded_context);
        let loss = lm_head_l2wrap_ce(logits, targets);

        NextTokenPredictionOutput { loss }
    }
}

impl<B> TrainStep for AutoRegressiveModel<B>
where
    B: AutodiffBackend + TrainBackend,
{
    type Input = AutoRegressiveBatch<B>;
    type Output = NextTokenPredictionOutput<B>;

    #[cfg(not(any(feature = "statetune", feature = "statepass")))]
    #[allow(unused)]
    fn step(&self, item: AutoRegressiveBatch<B>) -> TrainOutput<NextTokenPredictionOutput<B>> {
        let item = self.forward(item, None, None);
        let grads = item.loss.backward();

        TrainOutput::new(self, grads, item)
    }

    #[cfg(feature = "statetune")]
    fn step(&self, item: AutoRegressiveBatch<B>) -> TrainOutput<NextTokenPredictionOutput<B>> {
        let [batch_size, _] = item.inputs.dims();
        let state = self.state.get_state(batch_size);
        let item = self.forward(item, Some(state), None);
        let grads = item.loss.backward();

        TrainOutput::new(self, grads, item)
    }
}

impl<B> InferenceStep for AutoRegressiveModel<B>
where
    B: Backend + TrainBackend,
{
    type Input = AutoRegressiveBatch<B>;
    type Output = NextTokenPredictionOutput<B>;

    fn step(&self, item: AutoRegressiveBatch<B>) -> NextTokenPredictionOutput<B> {
        self.forward(item, None, None)
    }
}
