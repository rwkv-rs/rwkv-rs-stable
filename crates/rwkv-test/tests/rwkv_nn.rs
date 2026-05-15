#[cfg(feature = "cuda")]
mod cuda {
    use std::path::Path;

    use burn::{
        prelude::{Backend, Int, Tensor},
        tensor::{Device, TensorData},
    };
    use rwkv_test::{NumericTolerance, assert_values_close, read_safetensor_values};

    type B = burn::backend::Cuda<burn::tensor::bf16, i32>;
    type TestDevice = Device<B>;

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

    #[test]
    fn rwkv_lm_forward_loss_matches_trace() {
        let device = TestDevice::default();
        let model = rwkv_export::rwkv_lm_load_st::<B>(WEIGHTS, &device)
            .expect("load rwkv lm safetensors weights");
        let tokens = trace_tokens(&device);

        let output = model.forward(tokens.clone(), tokens, None, None);
        B::sync(&device).expect("sync rwkv lm forward");
        let actual = output
            .loss
            .into_data()
            .convert::<f32>()
            .iter::<f32>()
            .map(f64::from)
            .collect::<Vec<_>>();

        let (_dtype, shape, expected) = read_safetensor_values(
            Path::new(CASE_ROOT).join("loss/l2wrap_cross_entropy.safetensors"),
        )
        .expect("read trace loss");
        assert_eq!(shape, [1]);

        assert_values_close(
            "rwkv_nn/lm/forward/loss",
            &actual,
            &expected,
            NumericTolerance::new(2.0e-2, 1.0e-2, 0.999),
        );
    }

    fn trace_tokens(device: &TestDevice) -> Tensor<B, 2, Int> {
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
