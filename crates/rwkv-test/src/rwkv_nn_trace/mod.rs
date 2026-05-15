mod fixture;
mod safetensor_writer;
mod writer;

use std::fs;

use anyhow::{Context, Result, bail};
use burn::tensor::Device;

use crate::CompareRwkvNnArgs;
use self::{fixture::trace_tokens, writer::TraceWriter};

type TraceInnerBackend = burn::backend::Cuda<burn::tensor::bf16, i32>;
type TraceBackend = TraceInnerBackend;
type TraceDevice = Device<TraceBackend>;

const BATCH_SIZE: usize = 16;
const CONTEXT_LEN: usize = 512;

pub(crate) fn generate(args: &CompareRwkvNnArgs) -> Result<()> {
    if args.repeat == 0 {
        bail!("--repeat must be positive");
    }
    if args.actual.exists() {
        fs::remove_dir_all(&args.actual)
            .with_context(|| format!("failed to remove {}", args.actual.display()))?;
    }
    fs::create_dir_all(&args.actual)
        .with_context(|| format!("failed to create {}", args.actual.display()))?;

    let input_trace = "embedding/token_ids.safetensors";
    let input_source = args.baseline.join(input_trace);
    let input_target = args.actual.join(input_trace);
    let input_parent = input_target
        .parent()
        .expect("trace file has a parent directory");
    fs::create_dir_all(input_parent)
        .with_context(|| format!("failed to create {}", input_parent.display()))?;
    fs::copy(&input_source, &input_target).with_context(|| {
        format!(
            "failed to copy trace input {} to {}",
            input_source.display(),
            input_target.display()
        )
    })?;

    let device = TraceDevice::default();
    let model = rwkv_export::rwkv_lm_load_st::<TraceBackend>(&args.weights, &device)
        .with_context(|| format!("failed to load weights {}", args.weights.display()))?;
    let tokens = trace_tokens(&args.baseline, &device)?;

    let mut trace = TraceWriter::<TraceBackend>::new(&args.actual, &device);
    for index in 0..(args.warmup + args.repeat) {
        trace.collect_timing = index >= args.warmup;
        trace.write_outputs = false;
        trace.forward(&model, tokens.clone(), tokens.clone())?;
    }
    trace.collect_timing = false;
    trace.write_outputs = true;
    trace.forward(&model, tokens.clone(), tokens.clone())?;
    trace.flush_times(args.repeat, args.warmup)?;
    Ok(())
}
