use burn::backend::Autodiff;

#[cfg(feature = "cuda")]
pub type TestBackend = burn::backend::Cuda<f32, i32>;
#[cfg(all(not(feature = "cuda"), feature = "rocm"))]
pub type TestBackend = burn::backend::Rocm<f32, i32>;
#[cfg(all(not(feature = "cuda"), not(feature = "rocm"), feature = "vulkan"))]
pub type TestBackend = burn::backend::Vulkan<f32, i32>;
#[cfg(all(
    not(feature = "cuda"),
    not(feature = "rocm"),
    not(feature = "vulkan"),
    feature = "metal"
))]
pub type TestBackend = burn::backend::Metal<f32, i32>;
#[cfg(all(
    not(feature = "cuda"),
    not(feature = "rocm"),
    not(feature = "vulkan"),
    not(feature = "metal")
))]
pub type TestBackend = burn::backend::Cpu<f32, i32>;

pub type TestAutodiffBackend = Autodiff<TestBackend>;
