use rwkv_derive::ConfigBuilder;

#[derive(ConfigBuilder)]
#[config_builder(raw = "crate::", cell = "CONFIG")]
pub struct FinalConfig {
    value: usize,
}

fn main() {}
