#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]
#![warn(rustdoc::invalid_html_tags)]

//! Facade crate for RWKV workspace crates.

/// Re-exported Burn types and optional storage support used by generated code.
pub mod custom {
    pub use burn::*;
    #[cfg(feature = "store")]
    /// Re-exported Burn storage support.
    pub mod store {
        pub use burn_store::*;
    }
}

/// Imports the facade's re-exported Burn namespace as `burn`.
#[macro_export]
macro_rules! custom_mode {
    () => {
        use $crate::custom as burn;
    };
}

#[cfg(feature = "config")]
/// Configuration crate re-exports.
pub mod config {
    pub use rwkv_config::*;
}

#[cfg(feature = "data")]
/// Data processing crate re-exports.
pub mod data {
    pub use rwkv_data::*;
}

#[cfg(feature = "nn")]
/// Neural-network crate re-exports.
pub mod nn {
    pub use rwkv_nn::*;
}

#[cfg(feature = "train")]
/// Training crate re-exports.
pub mod train {
    pub use rwkv_train::*;
}

#[cfg(feature = "infer")]
/// Inference crate re-exports.
pub mod infer {
    pub use rwkv_infer::*;
}
