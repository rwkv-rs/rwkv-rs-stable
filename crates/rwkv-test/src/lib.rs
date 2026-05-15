#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]
#![warn(rustdoc::invalid_html_tags)]

//! RWKV trace comparison and `rwkv-nn` validation helpers.

mod cli;
mod compare;
mod display;
mod numeric;
mod safetensor;
mod timing;

#[cfg(feature = "cuda")]
mod rwkv_nn_trace;

pub use cli::{ColorMode, CompareArgs, CompareRwkvNnArgs, run_cli};
pub use compare::{compare, compare_rwkv_nn};
pub use numeric::{NumericStats, NumericTolerance, assert_values_close, numeric_stats};
pub use safetensor::read_safetensor_values;
