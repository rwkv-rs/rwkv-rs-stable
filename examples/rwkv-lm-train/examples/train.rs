#![recursion_limit = "256"]

use rwkv::{
    custom::tensor::backend::AutodiffBackend,
    nn::kernels::train::TrainBackend,
    train::learner::init::{BackendDeviceInit, init_cfg_paths, init_devices, init_log},
};

#[cfg(not(any(feature = "f32", feature = "flex32", feature = "f16")))]
#[allow(unused)]
type ElemType = rwkv::custom::tensor::bf16;
#[cfg(feature = "f32")]
type ElemType = f32;
#[cfg(feature = "flex32")]
type ElemType = rwkv::custom::tensor::flex32;
#[cfg(feature = "f16")]
type ElemType = rwkv::custom::tensor::f16;

pub fn launch<B: AutodiffBackend + BackendDeviceInit + TrainBackend>()
where
    <B as AutodiffBackend>::InnerBackend: BackendDeviceInit + TrainBackend,
{
    let (model_cfg_builder, mut train_cfg_builder) = init_cfg_paths(
        "examples/rwkv-lm-train/config/model.toml",
        "examples/rwkv-lm-train/config/train.toml",
    );

    let exp_log_path = init_log(&mut train_cfg_builder);
    let devices = init_devices::<B>(&train_cfg_builder);

    rwkv_lm_train::training::train::<B>(
        devices,
        model_cfg_builder,
        train_cfg_builder,
        &exp_log_path,
    );
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use rwkv::custom::backend::{Autodiff, Wgpu};

    use crate::{ElemType, launch};

    pub fn run() {
        launch::<Autodiff<Wgpu<ElemType, i32>>>();
    }
}

#[cfg(feature = "vulkan")]
mod vulkan {
    use rwkv::custom::backend::{
        Autodiff,
        Vulkan,
        autodiff::checkpoint::strategy::BalancedCheckpointing,
    };

    use crate::{ElemType, launch};

    pub fn run() {
        launch::<Autodiff<Vulkan<ElemType, i32>, BalancedCheckpointing>>();
    }
}

#[cfg(feature = "metal")]
mod metal {
    use rwkv::custom::backend::{Autodiff, Metal};

    use crate::{ElemType, launch};

    pub fn run() {
        launch::<Autodiff<Metal<ElemType, i32>>>();
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use rwkv::custom::backend::{
        Autodiff,
        Cuda,
        autodiff::checkpoint::strategy::BalancedCheckpointing,
    };

    use crate::{ElemType, launch};

    pub fn run() {
        launch::<Autodiff<Cuda<ElemType, i32>, BalancedCheckpointing>>();
    }
}

#[cfg(feature = "rocm")]
mod rocm {
    use rwkv::custom::backend::{
        Autodiff,
        Rocm,
        autodiff::checkpoint::strategy::BalancedCheckpointing,
    };

    use crate::{ElemType, launch};

    pub fn run() {
        launch::<Autodiff<Rocm<ElemType, i32>, BalancedCheckpointing>>();
    }
}

#[cfg(feature = "cpu")]
mod cpu {
    use rwkv::custom::backend::{Autodiff, Cpu};

    use crate::{ElemType, launch};

    pub fn run() {
        launch::<Autodiff<Cpu<ElemType, i32>>>();
    }
}

fn main() {
    #[cfg(feature = "wgpu")]
    wgpu::run();
    #[cfg(feature = "cuda")]
    cuda::run();
    #[cfg(feature = "rocm")]
    rocm::run();
    #[cfg(feature = "vulkan")]
    vulkan::run();
    #[cfg(feature = "metal")]
    metal::run();
    #[cfg(feature = "cpu")]
    cpu::run();
}
