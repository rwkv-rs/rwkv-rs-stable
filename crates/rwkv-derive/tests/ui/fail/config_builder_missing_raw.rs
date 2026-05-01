use rwkv_derive::ConfigBuilder;

#[derive(ConfigBuilder)]
#[config_builder(cell = "CONFIG")]
pub struct FinalConfig {
    value: usize,
}

fn main() {}
