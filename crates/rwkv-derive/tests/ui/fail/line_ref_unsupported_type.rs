use rwkv_derive::LineRef;

#[derive(LineRef)]
pub struct Sample {
    #[line_ref]
    tokens: Vec<u8>,
}

fn main() {}
