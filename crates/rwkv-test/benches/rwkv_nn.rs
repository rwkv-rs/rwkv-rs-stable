use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(feature = "cuda")]
mod cuda {
    use std::{hint::black_box, path::Path};

    use burn::{
        prelude::{Backend, Int, Tensor},
        tensor::{Device, TensorData},
    };
    use criterion::{BenchmarkId, Criterion};
    use rwkv_test::read_safetensor_values;

    type B = burn::backend::Cuda<burn::tensor::bf16, i32>;
    type BenchDevice = Device<B>;

    const WEIGHTS: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../weights/rwkv-init-0.1b-ctx512-test.st"
    );
    const CASE_ROOT: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/test_data/rwkv_lm/bf16/case_000000"
    );
    const BATCH_SIZE: usize = 16;
    const CONTEXT_LEN: usize = 512;

    pub(crate) fn rwkv_lm_forward(c: &mut Criterion) {
        let device = BenchDevice::default();
        let model = rwkv_export::rwkv_lm_load_st::<B>(WEIGHTS, &device)
            .expect("load rwkv lm safetensors weights");
        let tokens = trace_tokens(&device);
        let mut group = c.benchmark_group("rwkv-test/rwkv-nn/models/lm/forward");

        group.bench_with_input(BenchmarkId::new("loss", "trace"), &(), |bench, _| {
            bench.iter(|| {
                let output = model.forward(tokens.clone(), tokens.clone(), None, None);
                black_box(output.loss);
                B::sync(&device).expect("sync rwkv lm forward");
            });
        });

        group.finish();
    }

    fn trace_tokens(device: &BenchDevice) -> Tensor<B, 2, Int> {
        let (dtype, shape, values) =
            read_safetensor_values(Path::new(CASE_ROOT).join("embedding/token_ids.safetensors"))
                .expect("read trace token ids");
        assert_eq!(dtype.to_string(), "I64");
        assert_eq!(shape, [BATCH_SIZE, CONTEXT_LEN]);

        let tokens = values
            .into_iter()
            .map(|value| value as i32)
            .collect::<Vec<_>>();
        Tensor::from_ints(TensorData::new(tokens, [BATCH_SIZE, CONTEXT_LEN]), device)
    }
}

fn rwkv_lm_forward(c: &mut Criterion) {
    #[cfg(feature = "cuda")]
    cuda::rwkv_lm_forward(c);

    #[cfg(not(feature = "cuda"))]
    {
        let _ = c;
    }
}

criterion_group!(benches, rwkv_lm_forward);
criterion_main!(benches);
