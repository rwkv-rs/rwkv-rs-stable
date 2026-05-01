use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{Attribute, DeriveInput, Ident, LitStr, Path, Type};

use crate::ast;

pub(crate) fn expand(input: DeriveInput) -> syn::Result<TokenStream2> {
    let (name, fields) = ast::named_struct_fields(&input, "ConfigBuilder")?;
    let config_attrs = parse_config_builder_attrs(&input.attrs)?;
    let builder_name = format_ident!("{}Builder", name);
    let fields = collect_fields(fields);

    let builder_fields = fields.iter().map(|field| {
        let field_name = field.ident;
        let field_type = field.builder_type();

        quote! {
            #field_name: #field_type
        }
    });
    let set_methods = fields.iter().map(|field| {
        let field_name = field.ident;
        let method_name = format_ident!("set_{}", field_name);
        let field_type = field.builder_type();

        quote! {
            #[doc = concat!("Sets `", stringify!(#field_name), "`.")]
            pub fn #method_name(&mut self, val: #field_type) -> &mut Self {
                self.#field_name = val;
                self
            }
        }
    });
    let get_methods = fields.iter().map(|field| {
        let field_name = field.ident;
        let method_name = format_ident!("get_{}", field_name);
        let return_type = if let Some(inner_type) = field.option_inner {
            quote! { Option<#inner_type> }
        } else {
            let field_type = field.ty;
            quote! { Option<#field_type> }
        };

        quote! {
            #[doc = concat!("Returns the current `", stringify!(#field_name), "` value.")]
            pub fn #method_name(&self) -> #return_type {
                self.#field_name.clone()
            }
        }
    });
    let build_fields: Vec<_> = fields
        .iter()
        .map(|field| {
            let field_name = field.ident;

            if field.is_optional() {
                quote! {
                    #field_name: self.#field_name
                }
            } else {
                quote! {
                    #field_name: self
                        .#field_name
                        .expect(concat!("missing field: ", stringify!(#field_name)))
                }
            }
        })
        .collect();
    let build_local_fields = build_fields.clone();
    let load_from_raw_calls = fields.iter().map(|field| {
        let field_name = field.ident;
        let method_name = format_ident!("set_{}", field_name);

        if field.skip_raw {
            quote! {
                .#method_name(None)
            }
        } else {
            quote! {
                .#method_name(
                    crate::config_builder_helpers::IntoBuilderOption::into_builder_option(
                        raw.#field_name
                    )
                )
            }
        }
    });
    let raw_type = &config_attrs.raw_type;
    let cell_name = &config_attrs.cell_name;

    Ok(quote! {
        /// Builder for incrementally constructing the validated configuration.
        #[derive(Default)]
        pub struct #builder_name {
            #(#builder_fields,)*
        }

        impl #builder_name {
            /// Creates an empty builder.
            pub fn new() -> Self {
                Self::default()
            }

            #(#set_methods)*

            #(#get_methods)*

            /// Loads builder values from the raw configuration type.
            pub fn load_from_raw(raw: #raw_type) -> Self {
                let mut builder = Self::new();
                builder
                    #(#load_from_raw_calls)*;
                builder
            }

            /// Builds a local configuration instance without writing the global cell.
            pub fn build_local(self) -> std::sync::Arc<#name> {
                let config = #name {
                    #(#build_local_fields,)*
                };

                std::sync::Arc::new(config)
            }

            /// Builds the configuration instance and stores it in the global cell.
            pub fn build(self) -> std::sync::Arc<#name> {
                let config = #name {
                    #(#build_fields,)*
                };

                let arc_config = std::sync::Arc::new(config);
                #cell_name
                    .set(arc_config.clone())
                    .expect(concat!(stringify!(#cell_name), " already initialized"));

                arc_config
            }
        }
    })
}

fn collect_fields(fields: &syn::FieldsNamed) -> Vec<ConfigField<'_>> {
    fields
        .named
        .iter()
        .map(|field| {
            let ident = field
                .ident
                .as_ref()
                .expect("named struct fields must have identifiers");
            ConfigField {
                ident,
                ty: &field.ty,
                option_inner: ast::option_inner_type(&field.ty),
                skip_raw: ast::has_attr(&field.attrs, "skip_raw"),
            }
        })
        .collect()
}

fn parse_config_builder_attrs(attrs: &[Attribute]) -> syn::Result<ConfigBuilderAttrs> {
    let mut parsed = ConfigBuilderAttrsBuilder::default();

    for attr in attrs {
        if !attr.path().is_ident("config_builder") {
            continue;
        }

        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("raw") {
                let lit = meta.value()?.parse::<LitStr>()?;
                parsed.set_raw_type(&meta.path, lit)
            } else if meta.path.is_ident("cell") {
                let lit = meta.value()?.parse::<LitStr>()?;
                parsed.set_cell_name(&meta.path, lit)
            } else {
                Err(meta.error("unsupported config_builder parameter"))
            }
        })?;
    }

    parsed.finish()
}

#[derive(Default)]
struct ConfigBuilderAttrsBuilder {
    raw_type: Option<Path>,
    cell_name: Option<Path>,
}

impl ConfigBuilderAttrsBuilder {
    fn set_raw_type(&mut self, path: &Path, lit: LitStr) -> syn::Result<()> {
        if self.raw_type.is_some() {
            return Err(syn::Error::new_spanned(path, "duplicate `raw` parameter"));
        }

        self.raw_type = Some(parse_path_lit("raw", lit)?);
        Ok(())
    }

    fn set_cell_name(&mut self, path: &Path, lit: LitStr) -> syn::Result<()> {
        if self.cell_name.is_some() {
            return Err(syn::Error::new_spanned(path, "duplicate `cell` parameter"));
        }

        self.cell_name = Some(parse_path_lit("cell", lit)?);
        Ok(())
    }

    fn finish(self) -> syn::Result<ConfigBuilderAttrs> {
        Ok(ConfigBuilderAttrs {
            raw_type: self.raw_type.ok_or_else(|| {
                syn::Error::new(proc_macro2::Span::call_site(), "missing `raw` parameter")
            })?,
            cell_name: self.cell_name.ok_or_else(|| {
                syn::Error::new(proc_macro2::Span::call_site(), "missing `cell` parameter")
            })?,
        })
    }
}

fn parse_path_lit(name: &str, lit: LitStr) -> syn::Result<Path> {
    syn::parse_str::<Path>(&lit.value())
        .map_err(|err| syn::Error::new(lit.span(), format!("invalid `{name}` path: {err}")))
}

struct ConfigBuilderAttrs {
    raw_type: Path,
    cell_name: Path,
}

struct ConfigField<'a> {
    ident: &'a Ident,
    ty: &'a Type,
    option_inner: Option<&'a Type>,
    skip_raw: bool,
}

impl<'a> ConfigField<'a> {
    fn is_optional(&self) -> bool {
        self.option_inner.is_some()
    }

    fn builder_type(&self) -> TokenStream2 {
        let ty = self.ty;
        if self.is_optional() {
            quote! { #ty }
        } else {
            quote! { Option<#ty> }
        }
    }
}
