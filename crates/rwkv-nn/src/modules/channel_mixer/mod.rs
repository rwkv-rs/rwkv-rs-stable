use burn::{
    config::Config,
    module::{Module, Param},
    nn::{Linear, LinearConfig},
    prelude::{Backend, Int, Tensor},
    tensor::{TensorData, activation::relu},
};

use crate::{
    functions::init_weights::{get_token_shift_diff_scale, uniform_init, zeros_init},
    kernels::template::{
        addcmul::{AddcmulBackend, addcmul},
        token_shift_diff::{TokenShiftDiffBackend, token_shift_diff},
    },
};

#[derive(Config, Debug)]
pub struct ChannelMixerConfig {
    num_cells: usize,
    embedded_dim: usize,
}

impl ChannelMixerConfig {
    pub fn init<B: Backend>(&self, cell_id: usize, device: &B::Device) -> ChannelMixer<B> {
        ChannelMixer {
            key: LinearConfig::new(self.embedded_dim, self.embedded_dim * 4)
                .with_bias(false)
                .init(device),
            value: LinearConfig::new(self.embedded_dim * 4, self.embedded_dim)
                .with_bias(false)
                .init(device),
            token_shift_diff_scale: Param::from_tensor(Tensor::empty(
                [1, 1, self.embedded_dim],
                device,
            )),
            num_cells: self.num_cells,
            embedded_dim: self.embedded_dim,
            cell_id,
        }
    }
}

#[derive(Module, Debug)]

pub struct ChannelMixer<B: Backend> {
    pub key: Linear<B>,
    pub value: Linear<B>,
    pub token_shift_diff_scale: Param<Tensor<B, 3>>,

    pub num_cells: usize,
    pub embedded_dim: usize,

    #[module(skip)]
    cell_id: usize,
}

impl<B: Backend> ChannelMixer<B> {
    pub fn init_weights(&mut self, device: &B::Device) {
        let bound = 0.5 / (self.embedded_dim as f64).sqrt();

        uniform_init(&mut self.key.weight, -bound, bound);

        zeros_init(&mut self.value.weight);

        self.token_shift_diff_scale = Param::from_tensor(get_token_shift_diff_scale(
            self.num_cells,
            self.embedded_dim,
            self.cell_id,
            4.0,
            device,
        ));
    }
}

impl<B: AddcmulBackend + TokenShiftDiffBackend> ChannelMixer<B> {
    #[cfg_attr(
        feature = "trace",
        tracing::instrument(name = "rwkv.infer.model.channel_mixer", skip_all)
    )]
    pub fn forward(&self, channel_mixer_input: ChannelMixerIO<B>) -> ChannelMixerIO<B> {
        let should_return_token_shift = channel_mixer_input.embedded_token_shift.is_some();
        let ChannelMixerIO {
            embedded_context,
            embedded_token_shift,
        } = channel_mixer_input;
        let [batch_size, _context_len, embedded_dim] = embedded_context.dims();
        let device = embedded_context.device();
        let embedded_token_shift = embedded_token_shift
            .unwrap_or_else(|| Tensor::zeros([batch_size, embedded_dim], &device));
        let batch_ids = Tensor::<B, 1, Int>::from_ints(
            TensorData::new(
                (0..batch_size)
                    .map(|batch_index| batch_index as i32)
                    .collect::<Vec<_>>(),
                [batch_size],
            ),
            &device,
        );

        let token_shift_diff_output =
            token_shift_diff(embedded_context.clone(), embedded_token_shift, batch_ids);
        let embedded_context_shift = addcmul(
            embedded_context,
            token_shift_diff_output.token_shifted_diff,
            self.token_shift_diff_scale.val(),
        );

        let activated_key = relu(self.key.forward(embedded_context_shift)).powf_scalar(2.0);

        let value = self.value.forward(activated_key);

        ChannelMixerIO {
            embedded_context: value,
            embedded_token_shift: should_return_token_shift
                .then_some(token_shift_diff_output.next_token_shift),
        }
    }
}

pub struct ChannelMixerIO<B: Backend> {
    pub embedded_context: Tensor<B, 3>,
    pub embedded_token_shift: Option<Tensor<B, 2>>,
}
