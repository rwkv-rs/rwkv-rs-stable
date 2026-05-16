use burn::{
    config::Config,
    module::{Module, Param},
    nn::{GroupNorm, GroupNormConfig, Linear, LinearConfig},
    prelude::*,
};

use crate::{
    functions::init_weights::{constant_init, get_token_shift_diff_scale, zeros_init},
    kernels::train::time_mixer::gated_readout_combine::{
        GatedReadoutCombineBackend,
        gated_readout_combine,
        io::GatedReadoutCombineForwardInputs,
    },
    layers::lora::{ActivationFn, LoRA, LoRAConfig, LoRAType},
};

#[derive(Config, Debug)]
pub struct GatedReadoutConfig {
    num_cells: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,

    output_gate_lora_rank: usize,
}

impl GatedReadoutConfig {
    pub fn init<B: Backend>(&self, cell_id: usize, device: &B::Device) -> GatedReadout<B> {
        let empty_param = Param::from_tensor(Tensor::empty([1, 1, self.embedded_dim], device));

        let projection_output = LinearConfig::new(self.embedded_dim, self.embedded_dim)
            .with_bias(false)
            .init(device);

        GatedReadout {
            param_gate: empty_param,

            param_output_gate_lora: LoRAConfig::new(
                self.num_cells,
                self.embedded_dim,
                self.output_gate_lora_rank,
                self.head_size,
                false,
                ActivationFn::Sigmoid,
            )
            .init(device),

            param_receptance_key_bonus: Param::from_tensor(Tensor::empty(
                [self.num_heads, self.embedded_dim / self.num_heads],
                device,
            )),
            group_norm: GroupNormConfig::new(self.num_heads, self.embedded_dim)
                .with_epsilon(64e-5)
                .init(device),
            projection_output,

            num_cells: self.num_cells,
            embedded_dim: self.embedded_dim,
            num_heads: self.num_heads,
            head_size: self.head_size,

            cell_id,
        }
    }
}

#[derive(Module, Debug)]
pub struct GatedReadout<B: Backend> {
    param_gate: Param<Tensor<B, 3, Float>>,

    param_output_gate_lora: LoRA<B>,

    param_receptance_key_bonus: Param<Tensor<B, 2>>,
    group_norm: GroupNorm<B>,
    projection_output: Linear<B>,

    num_cells: usize,
    embedded_dim: usize,
    num_heads: usize,
    head_size: usize,

    #[module(skip)]
    cell_id: usize,
}

impl<B: Backend> GatedReadout<B> {
    pub fn init_weights(&mut self, device: &B::Device) {
        self.param_gate = Param::from_tensor(get_token_shift_diff_scale(
            self.num_cells,
            self.embedded_dim,
            self.cell_id,
            0.2,
            device,
        ));

        self.param_output_gate_lora
            .init_weight(self.cell_id, LoRAType::LearningRate, device);

        constant_init(&mut self.param_receptance_key_bonus, -0.04);

        zeros_init(&mut self.projection_output.weight);

        if let Some(ref mut gamma) = self.group_norm.gamma {
            let layer_scale = ((1 + self.cell_id) as f64 / self.num_cells as f64).powf(0.7);

            constant_init(gamma, layer_scale);
        }
    }

    #[cfg_attr(
        feature = "trace",
        tracing::instrument(name = "rwkv.infer.model.gated_readout", skip_all)
    )]
    pub fn forward(&self, gated_readout_input: GatedReadoutInput<B>) -> Tensor<B, 3>
    where
        B: GatedReadoutCombineBackend,
    {
        let GatedReadoutInput {
            gate_input,
            wkv7_forward_output,
            wkv7_forward_input_receptance,
            wkv7_forward_input_replacement_key,
            wkv7_forward_input_value,
        } = gated_readout_input;

        let gate = self.param_output_gate_lora.forward(gate_input);

        let norm_gamma = self
            .group_norm
            .gamma
            .as_ref()
            .expect("gated-readout group norm requires affine gamma")
            .val();
        let norm_beta = self
            .group_norm
            .beta
            .as_ref()
            .expect("gated-readout group norm requires affine beta")
            .val();

        let out_gated = gated_readout_combine(GatedReadoutCombineForwardInputs {
            wkv_output: wkv7_forward_output,
            norm_gamma,
            norm_beta,
            norm_epsilon: self.group_norm.epsilon,
            gate,
            receptance: wkv7_forward_input_receptance,
            replacement_key: wkv7_forward_input_replacement_key,
            value: wkv7_forward_input_value,
            bonus: self.param_receptance_key_bonus.val(),
        });

        self.projection_output.forward(out_gated)
    }

    pub(super) fn gate_scale(&self) -> Tensor<B, 3> {
        self.param_gate.val()
    }
}

pub struct GatedReadoutInput<B: Backend> {
    pub gate_input: Tensor<B, 3>,
    pub wkv7_forward_output: Tensor<B, 4>,
    pub wkv7_forward_input_receptance: Tensor<B, 4>,
    pub wkv7_forward_input_replacement_key: Tensor<B, 4>,
    pub wkv7_forward_input_value: Tensor<B, 4>,
}
