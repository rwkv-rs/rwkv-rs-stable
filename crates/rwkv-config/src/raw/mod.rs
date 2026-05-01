//! Raw configuration structures that mirror TOML input.

#[cfg(feature = "eval")]
/// Raw evaluation configuration.
pub mod eval;
#[cfg(feature = "infer")]
/// Raw inference configuration.
pub mod infer;
#[cfg(feature = "model")]
/// Raw model configuration.
pub mod model;
#[cfg(feature = "train")]
/// Raw training configuration.
pub mod train;

#[cfg(any(feature = "train", feature = "infer", feature = "eval"))]
#[macro_export]
/// Fills optional raw configuration fields with default values.
macro_rules! fill_default {
    ($s:expr, $( $field:ident : $value:expr ),+ $(,)?) => {
        $(
            if $s.$field.is_none() {
                $s.$field = Some($value);
            }
        )+
    };
}
