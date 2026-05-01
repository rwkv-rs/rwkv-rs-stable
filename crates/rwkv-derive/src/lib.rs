#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]
#![warn(rustdoc::invalid_html_tags)]

//! Procedural macros used by RWKV configuration and mmap data structures.

mod ast;
mod config_builder;
mod line_ref;

use proc_macro::TokenStream;
use syn::{DeriveInput, parse_macro_input};

/// Derives mmap line-reference serialization support for named structs.
#[proc_macro_derive(LineRef, attributes(line_ref))]
pub fn derive_line_ref(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    line_ref::expand(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// Derives a validated configuration builder for named structs.
#[proc_macro_derive(ConfigBuilder, attributes(config_builder, skip_raw))]
pub fn derive_config_builder(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    config_builder::expand(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
