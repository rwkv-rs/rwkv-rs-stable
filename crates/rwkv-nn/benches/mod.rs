#![allow(dead_code)]

use std::time::{Duration, Instant};

use burn::prelude::Backend;

#[cfg(feature = "cuda")]
pub type BenchBackend = burn::backend::Cuda<f32, i32>;
#[cfg(all(not(feature = "cuda"), feature = "rocm"))]
pub type BenchBackend = burn::backend::Rocm<f32, i32>;
#[cfg(all(not(feature = "cuda"), not(feature = "rocm"), feature = "vulkan"))]
pub type BenchBackend = burn::backend::Vulkan<f32, i32>;
#[cfg(all(
    not(feature = "cuda"),
    not(feature = "rocm"),
    not(feature = "vulkan"),
    feature = "metal"
))]
pub type BenchBackend = burn::backend::Metal<f32, i32>;
#[cfg(all(
    not(feature = "cuda"),
    not(feature = "rocm"),
    not(feature = "vulkan"),
    not(feature = "metal")
))]
pub type BenchBackend = burn::backend::Cpu<f32, i32>;

pub const BATCH_SIZES: [usize; 9] = [1, 2, 4, 8, 16, 32, 64, 128, 256];
pub const CONTEXT_LEN: usize = 512;
pub const EMBEDDED_DIM: usize = 4096;
const SUMMARY_MIN_DURATION: Duration = Duration::from_millis(200);
const SUMMARY_WARMUP_ITERS: usize = 3;

pub fn device<B: Backend>() -> B::Device {
    B::Device::default()
}

pub fn print_speedup_summary<B, Custom, Baseline>(
    name: &str,
    batch_size: usize,
    device: &B::Device,
    mut custom: Custom,
    mut baseline: Baseline,
) where
    B: Backend,
    Custom: FnMut(),
    Baseline: FnMut(),
{
    let custom_ns = measure_average_ns::<B, _>(device, &mut custom);
    let baseline_ns = measure_average_ns::<B, _>(device, &mut baseline);
    let speedup = baseline_ns / custom_ns;

    println!(
        "[speedup] {name}/{batch_size}: custom={:.3} ms baseline={:.3} ms speedup={speedup:.2}x",
        custom_ns / 1_000_000.0,
        baseline_ns / 1_000_000.0,
    );
}

fn measure_average_ns<B, Work>(device: &B::Device, work: &mut Work) -> f64
where
    B: Backend,
    Work: FnMut(),
{
    for _ in 0..SUMMARY_WARMUP_ITERS {
        work();
        B::sync(device).unwrap();
    }

    let mut iters = 1usize;

    loop {
        let start = Instant::now();

        for _ in 0..iters {
            work();
            B::sync(device).unwrap();
        }

        let elapsed = start.elapsed();
        if elapsed >= SUMMARY_MIN_DURATION || iters >= 32 {
            return elapsed.as_nanos() as f64 / iters as f64;
        }

        iters *= 2;
    }
}
