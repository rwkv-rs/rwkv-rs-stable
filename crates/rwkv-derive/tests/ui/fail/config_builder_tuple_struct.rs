use rwkv_derive::ConfigBuilder;

#[derive(ConfigBuilder)]
#[config_builder(raw = "crate::RawConfig", cell = "CONFIG")]
pub struct FinalConfig(usize);

fn main() {}
