use std::sync::OnceLock;

use rwkv_derive::ConfigBuilder;

pub mod config_builder_helpers {
    pub trait IntoBuilderOption<T> {
        fn into_builder_option(self) -> Option<T>;
    }

    impl<T> IntoBuilderOption<T> for T {
        fn into_builder_option(self) -> Option<T> {
            Some(self)
        }
    }

    impl<T> IntoBuilderOption<T> for Option<T> {
        fn into_builder_option(self) -> Option<T> {
            self
        }
    }
}

pub struct RawConfig {
    required: String,
    optional: Option<u32>,
}

#[derive(ConfigBuilder)]
#[config_builder(raw = "crate::RawConfig", cell = "CONFIG")]
pub struct FinalConfig {
    required: String,
    optional: Option<u32>,
    #[skip_raw]
    computed: usize,
}

pub static CONFIG: OnceLock<std::sync::Arc<FinalConfig>> = OnceLock::new();

fn main() {
    let raw = RawConfig {
        required: String::from("rwkv"),
        optional: Some(7),
    };

    let mut builder = FinalConfigBuilder::load_from_raw(raw);
    assert_eq!(builder.get_required().as_deref(), Some("rwkv"));
    assert_eq!(builder.get_optional(), Some(7));
    assert_eq!(builder.get_computed(), None);

    builder.set_computed(Some(42));
    let config = builder.build_local();
    assert_eq!(config.required, "rwkv");
    assert_eq!(config.optional, Some(7));
    assert_eq!(config.computed, 42);
}
