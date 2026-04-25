#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]
#![warn(rustdoc::invalid_html_tags)]

pub mod custom {
    pub use burn::*;
    #[cfg(feature = "store")]
    pub mod store {
        pub use burn_store::*;
    }
}

#[macro_export]
macro_rules! custom_mode {
    () => {
        use $crate::custom as burn;
    };
}

#[cfg(feature = "config")]
pub mod config {
    pub use rwkv_config::*;
}

#[cfg(feature = "data")]
pub mod data {
    pub use rwkv_data::*;
}

#[cfg(feature = "nn")]
pub mod nn {
    pub use rwkv_nn::*;
}

#[cfg(feature = "train")]
pub mod train {
    pub use rwkv_train::*;
}

#[cfg(feature = "infer")]
pub mod infer {
    pub use rwkv_infer::*;
}
