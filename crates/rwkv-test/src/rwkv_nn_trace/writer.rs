use std::{collections::HashMap, fs, path::Path, time::Instant};

use anyhow::{Context, Result, bail};
use burn::{
    prelude::{Backend, Int, Tensor},
    tensor::{DType, Element},
};
use rwkv_nn::{
    kernels::train::{TrainBackend, lm_head_l2wrap_ce::lm_head_l2wrap_ce},
    models::lm::RwkvLM,
    modules::{channel_mixer::ChannelMixerIO, time_mixer::TimeMixerIO},
};

use super::safetensor_writer::write_safetensor;

pub(super) struct TraceWriter<'a, B: Backend> {
    root: &'a Path,
    device: &'a B::Device,
    timings: HashMap<String, Vec<u64>>,
    pub(super) collect_timing: bool,
    pub(super) write_outputs: bool,
}

impl<'a, B> TraceWriter<'a, B>
where
    B: TrainBackend,
{
    pub(super) fn new(root: &'a Path, device: &'a B::Device) -> Self {
        Self {
            root,
            device,
            timings: HashMap::new(),
            collect_timing: false,
            write_outputs: false,
        }
    }

    pub(super) fn forward(
        &mut self,
        model: &RwkvLM<B>,
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
    ) -> Result<()> {
        let embedded_context =
            self.trace_float("embedding", "embedding/embedded_context", || {
                model.embed.forward(inputs)
            })?;
        let embedded_context =
            self.trace_float("layer_norm0", "layer_norm0/embedded_context", || {
                model.layer_norm_for_first_cell.forward(embedded_context)
            })?;
        let embedded_context = self.forward_cells(model, embedded_context)?;
        let embedded_context = self.trace_float("lm_head", "lm_head/embedded_context", || {
            model.layer_norm_for_unembed.forward(embedded_context)
        })?;
        let logits = model.unembed.forward(embedded_context);
        let loss = self.trace_float_as(
            "loss/l2wrap_cross_entropy",
            "loss/l2wrap_cross_entropy",
            DType::F32,
            || lm_head_l2wrap_ce(logits, targets),
        )?;
        drop(loss);
        Ok(())
    }

    fn forward_cells(
        &mut self,
        model: &RwkvLM<B>,
        embedded_context: Tensor<B, 3>,
    ) -> Result<Tensor<B, 3>> {
        let mut value_from_first_cell = Tensor::zeros_like(&embedded_context);
        let mut embedded_context = embedded_context;

        for (cell_index, cell) in model.cells.cells.iter().enumerate() {
            let prefix = format!("cells/cell_{cell_index:04}");
            let embedded_context_before_cell = embedded_context;

            let embedded_context_normalized =
                self.trace_float_timing(&format!("{prefix}/pre_layer_norm_for_time_mix"), || {
                    cell.pre_layer_norm_for_time_mix
                        .forward(embedded_context_before_cell.clone())
                })?;
            let time_mixer_input = TimeMixerIO {
                embedded_context: embedded_context_normalized,
                value_from_first_cell: value_from_first_cell.clone(),
                state: None,
            };
            let time_mixer_output = self
                .trace_time_mixer(&format!("{prefix}/time_mixer"), || {
                    cell.time_mixer.forward(time_mixer_input)
                })?;
            embedded_context = self.trace_float(
                &format!("{prefix}/embedded_context_after_time_mixer"),
                &format!("{prefix}/embedded_context_after_time_mixer"),
                || embedded_context_before_cell + time_mixer_output.embedded_context,
            )?;
            value_from_first_cell = time_mixer_output.value_from_first_cell;

            let embedded_context_normalized = self.trace_float_timing(
                &format!("{prefix}/pre_layer_norm_for_channel_mix"),
                || {
                    cell.pre_layer_norm_for_channel_mix
                        .forward(embedded_context.clone())
                },
            )?;
            let channel_mixer_input = ChannelMixerIO {
                embedded_context: embedded_context_normalized,
                embedded_token_shift: None,
            };
            let channel_mixer_output = self
                .trace_channel_mixer(&format!("{prefix}/channel_mixer"), || {
                    cell.channel_mixer.forward(channel_mixer_input)
                })?;
            embedded_context = self.trace_float(
                &format!("{prefix}/embedded_context_after_channel_mixer"),
                &format!("{prefix}/embedded_context_after_channel_mixer"),
                || embedded_context + channel_mixer_output.embedded_context,
            )?;
        }

        Ok(embedded_context)
    }

    fn trace_time_mixer(
        &mut self,
        prefix: &str,
        work: impl FnOnce() -> TimeMixerIO<B>,
    ) -> Result<TimeMixerIO<B>> {
        let start = Instant::now();
        let output = work();
        B::sync(self.device).context("failed to sync timed time mixer")?;
        let elapsed_ns = start.elapsed().as_nanos() as u64;
        self.write_time(prefix, elapsed_ns)?;
        if self.write_outputs {
            self.write_float(
                &format!("{prefix}/embedded_context"),
                output.embedded_context.clone(),
            )?;
            self.write_float(
                &format!("{prefix}/value_from_first_cell"),
                output.value_from_first_cell.clone(),
            )?;
        }
        Ok(output)
    }

    fn trace_channel_mixer(
        &mut self,
        prefix: &str,
        work: impl FnOnce() -> ChannelMixerIO<B>,
    ) -> Result<ChannelMixerIO<B>> {
        let start = Instant::now();
        let output = work();
        B::sync(self.device).context("failed to sync timed channel mixer")?;
        let elapsed_ns = start.elapsed().as_nanos() as u64;
        self.write_time(prefix, elapsed_ns)?;
        if self.write_outputs {
            self.write_float(
                &format!("{prefix}/embedded_context"),
                output.embedded_context.clone(),
            )?;
        }
        Ok(output)
    }

    fn trace_float<const D: usize>(
        &mut self,
        module: &str,
        name: &str,
        work: impl FnOnce() -> Tensor<B, D>,
    ) -> Result<Tensor<B, D>> {
        self.trace_float_as(module, name, B::FloatElem::dtype(), work)
    }

    fn trace_float_as<const D: usize>(
        &mut self,
        module: &str,
        name: &str,
        dtype: DType,
        work: impl FnOnce() -> Tensor<B, D>,
    ) -> Result<Tensor<B, D>> {
        let start = Instant::now();
        let output = work();
        B::sync(self.device).with_context(|| format!("failed to sync timed trace {name}"))?;
        let elapsed_ns = start.elapsed().as_nanos() as u64;
        self.write_time(module, elapsed_ns)?;
        if self.write_outputs {
            self.write_float_as(name, output.clone(), dtype)?;
        }
        Ok(output)
    }

    fn trace_float_timing<const D: usize>(
        &mut self,
        module: &str,
        work: impl FnOnce() -> Tensor<B, D>,
    ) -> Result<Tensor<B, D>> {
        let start = Instant::now();
        let output = work();
        B::sync(self.device).with_context(|| format!("failed to sync timed trace {module}"))?;
        let elapsed_ns = start.elapsed().as_nanos() as u64;
        self.write_time(module, elapsed_ns)?;
        Ok(output)
    }

    fn write_float<const D: usize>(&self, name: &str, tensor: Tensor<B, D>) -> Result<()> {
        self.write_float_as(name, tensor, B::FloatElem::dtype())
    }

    fn write_float_as<const D: usize>(
        &self,
        name: &str,
        tensor: Tensor<B, D>,
        dtype: DType,
    ) -> Result<()> {
        let data = tensor.into_data().convert_dtype(dtype);
        write_safetensor(&self.root.join(format!("{name}.safetensors")), data)
    }

    fn write_time(&mut self, module: &str, elapsed_ns: u64) -> Result<()> {
        if elapsed_ns == 0 {
            bail!("{module} timing must be positive");
        }
        if self.collect_timing {
            self.timings
                .entry(module.to_owned())
                .or_default()
                .push(elapsed_ns);
        }
        Ok(())
    }

    pub(super) fn flush_times(&self, repeat: usize, warmup: usize) -> Result<()> {
        for (module, samples) in &self.timings {
            if samples.len() != repeat {
                bail!(
                    "{module} collected {} timing samples, expected {repeat}",
                    samples.len()
                );
            }
            let elapsed_ns =
                (samples.iter().sum::<u64>() as f64 / samples.len() as f64).round() as u64;
            let samples_json = samples
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(",");
            self.write_time_record(module, elapsed_ns, repeat, warmup, &samples_json)?;
        }
        Ok(())
    }

    fn write_time_record(
        &self,
        module: &str,
        elapsed_ns: u64,
        repeat: usize,
        warmup: usize,
        samples_json: &str,
    ) -> Result<()> {
        let path = self.root.join(format!("timing/{module}.time.json"));
        let parent = path.parent().expect("trace file has a parent directory");
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
        fs::write(
            &path,
            format!("{{\"module\":\"{module}\",\"elapsed_ns\":{elapsed_ns},\"repeat\":{repeat},\"warmup\":{warmup},\"samples_ns\":[{samples_json}]}}\n"),
        )
        .with_context(|| format!("failed to write {}", path.display()))
    }
}
