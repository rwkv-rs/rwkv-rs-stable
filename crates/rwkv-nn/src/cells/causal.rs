use burn::{
    config::Config,
    module::Module,
    nn::{LayerNorm, LayerNormConfig},
    prelude::{Backend, Int, Tensor},
};

use crate::{
    kernels::{
        template::{addcmul::AddcmulBackend, token_shift_diff::TokenShiftDiffBackend},
        train::time_mixer::{mix6::Mix6Backend, wkv7::Wkv7Backend},
    },
    modules::{
        channel_mixer::{ChannelMixer, ChannelMixerConfig, ChannelMixerIO},
        time_mixer::{TimeMixer, TimeMixerConfig, TimeMixerIO},
    },
};

#[derive(Config, Debug)]
pub struct MultiCausalCellsConfig {
    num_cells: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,
}

impl MultiCausalCellsConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiCausalCells<B> {
        MultiCausalCells {
            cells: (0..self.num_cells)
                .map(|i| {
                    CausalCellConfig::new(
                        self.num_cells,
                        self.embedded_dim,
                        self.num_heads,
                        self.head_size,
                    )
                    .init(i, device)
                })
                .collect(),

            num_cells: self.num_cells,
            embedded_dim: self.embedded_dim,
            num_heads: self.num_heads,
            head_size: self.head_size,
        }
    }
}

#[derive(Module, Debug)]
pub struct MultiCausalCells<B: Backend> {
    pub cells: Vec<CausalCell<B>>,

    num_cells: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,
}

impl<B: Backend> MultiCausalCells<B> {
    pub fn init_weights(&mut self, device: &B::Device) {
        self.cells
            .iter_mut()
            .for_each(|cell| cell.init_weights(device));
    }

    #[cfg_attr(
        feature = "trace",
        tracing::instrument(name = "rwkv.infer.model.cells", skip_all)
    )]
    pub fn forward(&self, multi_causal_cells_input: MultiCausalCellsIO<B>) -> MultiCausalCellsIO<B>
    where
        B: AddcmulBackend + TokenShiftDiffBackend + Mix6Backend + Wkv7Backend,
    {
        let MultiCausalCellsIO {
            embedded_context,
            mut state,
            mut embedded_token_shift_for_channel_mix,
        } = multi_causal_cells_input;

        let mut value_from_first_cell = Tensor::zeros_like(&embedded_context);
        let mut embedded_context = embedded_context;

        if let Some(v) = state.as_ref() {
            debug_assert_eq!(v.len(), self.num_cells);
        }
        if let Some(v) = embedded_token_shift_for_channel_mix.as_ref() {
            debug_assert_eq!(v.len(), self.num_cells);
        }

        for (cell_id, cell) in self.cells.iter().enumerate() {
            let state_of_the_cell = state.as_ref().map(|v| v[cell_id].clone());
            let cell_channel_shift = embedded_token_shift_for_channel_mix
                .as_ref()
                .map(|v| v[cell_id].clone());

            let CausalCellIO {
                embedded_context: next_embedded_context,
                value_from_first_cell: next_value_from_first_cell,
                state: next_state,
                embedded_token_shift_for_channel_mix: next_embedded_token_shift_for_channel_mix,
            } = cell.forward(CausalCellIO {
                embedded_context,
                value_from_first_cell,
                state: state_of_the_cell,
                embedded_token_shift_for_channel_mix: cell_channel_shift,
            });

            embedded_context = next_embedded_context;
            value_from_first_cell = next_value_from_first_cell;

            if let Some(v) = state.as_mut() {
                v[cell_id] = next_state.unwrap();
            }

            if let Some(v) = embedded_token_shift_for_channel_mix.as_mut() {
                v[cell_id] = next_embedded_token_shift_for_channel_mix.unwrap();
            }
        }

        MultiCausalCellsIO {
            embedded_context,
            state,
            embedded_token_shift_for_channel_mix,
        }
    }
}

pub struct MultiCausalCellsIO<B: Backend> {
    pub embedded_context: Tensor<B, 3>, // [batch_size, context_length, embedded_dim]
    pub state: Option<Vec<Tensor<B, 4>>>, // num_cells [batch_size, num_heads, head_size, head_size]
    pub embedded_token_shift_for_channel_mix: Option<Vec<Tensor<B, 2>>>, // num_cells [batch_size, embedded_dim]
}

pub struct MultiCausalCellsInferIO<B: Backend> {
    pub embedded_tokens: Tensor<B, 2>,    // [num_tokens, embedded_dim]
    pub cursors: Tensor<B, 1, Int>,       // [batch_size]
    pub state: Option<Vec<Tensor<B, 4>>>, // num_cells [batch_size, num_heads, head_size, head_size]
    pub embedded_token_shift_for_channel_mix: Option<Vec<Tensor<B, 2>>>, // num_cells [batch_size, embedded_dim]
}

#[derive(Config, Debug)]
pub struct CausalCellConfig {
    num_cells: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,
}

impl CausalCellConfig {
    pub fn init<B: Backend>(&self, cell_id: usize, device: &B::Device) -> CausalCell<B> {
        CausalCell {
            pre_layer_norm_for_time_mix: LayerNormConfig::new(self.embedded_dim).init(device),
            time_mixer: TimeMixerConfig::new(
                self.num_cells,
                self.embedded_dim,
                self.num_heads,
                self.head_size,
            )
            .init(cell_id, device),
            pre_layer_norm_for_channel_mix: LayerNormConfig::new(self.embedded_dim).init(device),
            channel_mixer: ChannelMixerConfig::new(self.num_cells, self.embedded_dim)
                .init(cell_id, device),
            cell_id,
        }
    }
}

#[derive(Module, Debug)]
pub struct CausalCell<B: Backend> {
    pub pre_layer_norm_for_time_mix: LayerNorm<B>,
    pub pre_layer_norm_for_channel_mix: LayerNorm<B>,
    pub time_mixer: TimeMixer<B>,
    pub channel_mixer: ChannelMixer<B>,

    #[module(skip)]
    cell_id: usize,
}

impl<B: Backend> CausalCell<B> {
    pub fn init_weights(&mut self, device: &B::Device) {
        self.time_mixer.init_weights(device);
        self.channel_mixer.init_weights(device);
    }
}

impl<B: AddcmulBackend + TokenShiftDiffBackend + Mix6Backend + Wkv7Backend> CausalCell<B> {
    #[cfg_attr(feature = "trace", tracing::instrument(name = "rwkv.infer.model.cell", skip_all, fields(cell_id = self.cell_id)))]
    pub fn forward(&self, causal_cell_input: CausalCellIO<B>) -> CausalCellIO<B> {
        let embedded_context = causal_cell_input.embedded_context;

        let embedded_context_normalized = self
            .pre_layer_norm_for_time_mix
            .forward(embedded_context.clone());
        let time_mixer_input = TimeMixerIO {
            embedded_context: embedded_context_normalized,
            value_from_first_cell: causal_cell_input.value_from_first_cell.clone(),
            state: causal_cell_input.state,
        };

        let time_mixer_output = self.time_mixer.forward(time_mixer_input);

        let embedded_context = embedded_context + time_mixer_output.embedded_context;

        let embedded_context_normalized = self
            .pre_layer_norm_for_channel_mix
            .forward(embedded_context.clone());

        let channel_mixer_input = ChannelMixerIO {
            embedded_context: embedded_context_normalized,
            embedded_token_shift: causal_cell_input.embedded_token_shift_for_channel_mix,
        };

        let channel_mixer_output = self.channel_mixer.forward(channel_mixer_input);

        let embedded_context = embedded_context + channel_mixer_output.embedded_context;

        CausalCellIO {
            embedded_context,
            value_from_first_cell: time_mixer_output.value_from_first_cell,
            state: time_mixer_output.state,
            embedded_token_shift_for_channel_mix: channel_mixer_output.embedded_token_shift,
        }
    }
}

pub struct CausalCellIO<B: Backend> {
    pub embedded_context: Tensor<B, 3>, // [batch_size, context_len, embedded_dim]
    pub value_from_first_cell: Tensor<B, 3>, // [batch_size, context_len, embedded_dim]
    pub state: Option<Tensor<B, 4>>,    // [batch_size, num_heads, head_size, head_size]
    pub embedded_token_shift_for_channel_mix: Option<Tensor<B, 2>>, // [batch_size, embedded_dim]
}
