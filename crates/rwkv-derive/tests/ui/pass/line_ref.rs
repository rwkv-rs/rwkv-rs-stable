use rwkv_data::mmap::idx::LineRefSample;
use rwkv_derive::LineRef;

mod rwkv_data {
    pub mod mmap {
        pub mod bin {
            use std::{borrow::Cow, marker::PhantomData};

            pub struct BinReader<T> {
                _marker: PhantomData<T>,
            }

            impl Default for BinReader<u8> {
                fn default() -> Self {
                    Self {
                        _marker: PhantomData,
                    }
                }
            }

            impl BinReader<u8> {
                pub fn get(&self, _offset: u64, length: u64) -> Cow<'static, [u8]> {
                    if length == 16 {
                        Cow::Owned(1_u128.to_le_bytes().to_vec())
                    } else {
                        Cow::Borrowed(b"rwkv")
                    }
                }
            }
        }

        pub mod idx {
            pub trait LineRefSample: Sized {
                type Serialized;

                fn to_serialized(&self, map: &super::map::Map) -> Self::Serialized;

                fn from_serialized(
                    data: &Self::Serialized,
                    bin: &super::bin::BinReader<u8>,
                ) -> Self;
            }
        }

        pub mod map {
            pub struct Map;

            impl Map {
                pub fn get_with_str(&self, value: &str) -> (u64, u64) {
                    (0, value.len() as u64)
                }

                pub fn get_with_u128(&self, _value: u128) -> (u64, u64) {
                    (0, 16)
                }
            }
        }
    }
}

#[derive(LineRef)]
pub struct Sample {
    #[line_ref]
    text: String,
    #[line_ref]
    token: u128,
}

fn main() {
    let sample = Sample {
        text: String::from("rwkv"),
        token: 1,
    };
    let map = rwkv_data::mmap::map::Map;
    let serialized = sample.to_serialized(&map);
    let bin = rwkv_data::mmap::bin::BinReader::default();
    let restored = Sample::from_serialized(&serialized, &bin);

    assert_eq!(
        Sample::serialized_size(),
        std::mem::size_of::<SampleSerialized>()
    );
    assert_eq!(restored.text, "rwkv");
    assert_eq!(restored.token, 1);
}
