use burn_cubecl::{CubeRuntime, cubecl::ir::HardwareProperties, tensor::CubeTensor};
use serde::{Deserialize, Serialize};

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CubeHardwareFingerprint {
    pub(crate) load_width: u32,
    pub(crate) plane_size: u32,
    pub(crate) max_units_per_cube: u32,
    pub(crate) max_cube_dim: (u32, u32, u32),
    pub(crate) max_shared_memory_size: usize,
    pub(crate) max_vector_size: usize,
    pub(crate) num_streaming_multiprocessors: Option<u32>,
    pub(crate) num_tensor_cores: Option<u32>,
    pub(crate) min_tensor_cores_dim: Option<u32>,
}

impl CubeHardwareFingerprint {
    pub(crate) fn from_hardware(hardware: &HardwareProperties) -> Self {
        Self {
            load_width: hardware.load_width,
            plane_size: hardware.plane_size_max,
            max_units_per_cube: hardware.max_units_per_cube,
            max_cube_dim: hardware.max_cube_dim,
            max_shared_memory_size: hardware.max_shared_memory_size,
            max_vector_size: hardware.max_vector_size,
            num_streaming_multiprocessors: hardware.num_streaming_multiprocessors,
            num_tensor_cores: hardware.num_tensor_cores,
            min_tensor_cores_dim: hardware.min_tensor_cores_dim,
        }
    }
}

impl core::fmt::Display for CubeHardwareFingerprint {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "load{}:p{}:u{}:cube{:?}:s{}:vec{}:sm{:?}:tc{:?}:tcdim{:?}",
            self.load_width,
            self.plane_size,
            self.max_units_per_cube,
            self.max_cube_dim,
            self.max_shared_memory_size,
            self.max_vector_size,
            self.num_streaming_multiprocessors,
            self.num_tensor_cores,
            self.min_tensor_cores_dim
        )
    }
}

pub(crate) fn assert_linear_readable<R: CubeRuntime>(name: &str, tensor: &CubeTensor<R>) {
    if tensor.meta.shape().num_elements() == 0 {
        return;
    }

    assert!(
        R::can_read_tensor(tensor.meta.shape(), tensor.meta.strides()),
        "{name} must use a runtime-readable linear layout, got shape {:?}, strides {:?}",
        tensor.meta.shape(),
        tensor.meta.strides()
    );
}
