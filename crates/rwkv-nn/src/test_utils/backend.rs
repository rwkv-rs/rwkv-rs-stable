use burn::backend::Autodiff;

/// CUDA backend used by kernel tests when the `cuda` feature is enabled.
#[cfg(feature = "cuda")]
pub type TestBackend = burn::backend::Cuda<f32, i32>;
/// ROCm backend used by kernel tests when the `rocm` feature is enabled.
#[cfg(all(not(feature = "cuda"), feature = "rocm"))]
pub type TestBackend = burn::backend::Rocm<f32, i32>;
/// Vulkan backend used by kernel tests when the `vulkan` feature is enabled.
#[cfg(all(not(feature = "cuda"), not(feature = "rocm"), feature = "vulkan"))]
pub type TestBackend = burn::backend::Vulkan<f32, i32>;
/// Metal backend used by kernel tests when the `metal` feature is enabled.
#[cfg(all(
    not(feature = "cuda"),
    not(feature = "rocm"),
    not(feature = "vulkan"),
    feature = "metal"
))]
pub type TestBackend = burn::backend::Metal<f32, i32>;
/// CPU backend used by kernel tests when no GPU backend feature is enabled.
#[cfg(all(
    not(feature = "cuda"),
    not(feature = "rocm"),
    not(feature = "vulkan"),
    not(feature = "metal")
))]
pub type TestBackend = burn::backend::Cpu<f32, i32>;

/// Autodiff wrapper for the default kernel test backend.
pub type TestAutodiffBackend = Autodiff<TestBackend>;
/// Device type for the default kernel test backend.
pub type TestDevice = burn::tensor::Device<TestBackend>;
/// Device type for the autodiff kernel test backend.
pub type TestAutodiffDevice = burn::tensor::Device<TestAutodiffBackend>;
