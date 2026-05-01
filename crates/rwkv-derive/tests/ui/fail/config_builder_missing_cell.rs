use rwkv_derive::ConfigBuilder;

#[derive(ConfigBuilder)]
#[config_builder(raw = "crate::RawConfig")]
pub struct FinalConfig {
    value: usize,
}

fn main() {}
