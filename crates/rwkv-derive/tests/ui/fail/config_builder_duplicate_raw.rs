use rwkv_derive::ConfigBuilder;

#[derive(ConfigBuilder)]
#[config_builder(
    raw = "crate::RawConfig",
    raw = "crate::OtherRawConfig",
    cell = "CONFIG"
)]
pub struct FinalConfig {
    value: usize,
}

fn main() {}
