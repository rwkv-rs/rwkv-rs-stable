use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{DeriveInput, Ident, Type};

use crate::ast;

pub(crate) fn expand(input: DeriveInput) -> syn::Result<TokenStream2> {
    let (name, fields) = ast::named_struct_fields(&input, "LineRef")?;
    let mut fields = collect_fields(fields)?;

    fields.sort_by_key(|field| field.ident().to_string());

    let serialized_name = format_ident!("{}Serialized", name);
    let mut serialized_fields = Vec::new();
    let mut to_serialized_lets = Vec::new();
    let mut to_serialized_fields = Vec::new();
    let mut from_serialized_assignments = Vec::new();

    for field in fields {
        match field {
            SerializedField::LineRef(field) => {
                let field_name = field.ident;
                let offset_field = format_ident!("{}_offset", field_name);
                let length_field = format_ident!("{}_length", field_name);

                serialized_fields.push(quote! { #offset_field: u64 });
                serialized_fields.push(quote! { #length_field: u64 });
                to_serialized_fields.push(quote! { #offset_field });
                to_serialized_fields.push(quote! { #length_field });

                match field.kind {
                    LineRefKind::String => {
                        to_serialized_lets.push(quote! {
                            let (#offset_field, #length_field) =
                                map.get_with_str(&self.#field_name);
                        });
                        from_serialized_assignments.push(quote! {
                            #field_name: {
                                let tokens = bin.get(data.#offset_field, data.#length_field);
                                String::from_utf8(tokens.into_owned()).unwrap()
                            }
                        });
                    }
                    LineRefKind::U128 => {
                        to_serialized_lets.push(quote! {
                            let (#offset_field, #length_field) =
                                map.get_with_u128(self.#field_name);
                        });
                        from_serialized_assignments.push(quote! {
                            #field_name: {
                                let tokens = bin.get(data.#offset_field, data.#length_field);
                                let slice = tokens.as_ref();
                                let bytes: [u8; 16] = slice.try_into().unwrap();
                                u128::from_le_bytes(bytes)
                            }
                        });
                    }
                }
            }
            SerializedField::Plain(field) => {
                let field_name = field.ident;
                let field_type = field.ty;

                serialized_fields.push(quote! { #field_name: #field_type });
                to_serialized_fields.push(quote! { #field_name: self.#field_name });
                from_serialized_assignments.push(quote! { #field_name: data.#field_name });
            }
        }
    }

    Ok(quote! {
        /// Serialized mmap layout for this line-reference sample.
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        #[repr(C)]
        pub struct #serialized_name {
            #(#serialized_fields,)*
        }

        impl #name {
            /// Returns the byte size of the serialized mmap layout.
            pub fn serialized_size() -> usize {
                std::mem::size_of::<#serialized_name>()
            }
        }

        impl rwkv_data::mmap::idx::LineRefSample for #name {
            type Serialized = #serialized_name;

            fn to_serialized(&self, map: &rwkv_data::mmap::map::Map) -> Self::Serialized {
                #(#to_serialized_lets)*

                #serialized_name {
                    #(#to_serialized_fields,)*
                }
            }

            fn from_serialized(
                data: &Self::Serialized,
                bin: &rwkv_data::mmap::bin::BinReader<u8>,
            ) -> Self {
                Self {
                    #(#from_serialized_assignments,)*
                }
            }
        }
    })
}

fn collect_fields(fields: &syn::FieldsNamed) -> syn::Result<Vec<SerializedField<'_>>> {
    let mut parsed = Vec::with_capacity(fields.named.len());
    let mut errors: Option<syn::Error> = None;

    for field in &fields.named {
        let ident = field
            .ident
            .as_ref()
            .expect("named struct fields must have identifiers");

        let parsed_field = if ast::has_attr(&field.attrs, "line_ref") {
            line_ref_kind(&field.ty)
                .map(|kind| SerializedField::LineRef(LineRefField { ident, kind }))
        } else {
            Ok(SerializedField::Plain(PlainField {
                ident,
                ty: &field.ty,
            }))
        };

        match parsed_field {
            Ok(parsed_field) => parsed.push(parsed_field),
            Err(error) => {
                if let Some(errors) = &mut errors {
                    errors.combine(error);
                } else {
                    errors = Some(error);
                }
            }
        }
    }

    if let Some(errors) = errors {
        return Err(errors);
    }

    Ok(parsed)
}

fn line_ref_kind(ty: &Type) -> syn::Result<LineRefKind> {
    let Type::Path(type_path) = ty else {
        return Err(syn::Error::new_spanned(
            ty,
            "line_ref only supports String and u128",
        ));
    };

    let Some(last_segment) = type_path.path.segments.last() else {
        return Err(syn::Error::new_spanned(
            ty,
            "line_ref only supports String and u128",
        ));
    };

    match last_segment.ident.to_string().as_str() {
        "String" => Ok(LineRefKind::String),
        "u128" => Ok(LineRefKind::U128),
        _ => Err(syn::Error::new_spanned(
            ty,
            "line_ref only supports String and u128",
        )),
    }
}

enum SerializedField<'a> {
    LineRef(LineRefField<'a>),
    Plain(PlainField<'a>),
}

impl SerializedField<'_> {
    fn ident(&self) -> &Ident {
        match self {
            SerializedField::LineRef(field) => field.ident,
            SerializedField::Plain(field) => field.ident,
        }
    }
}

struct LineRefField<'a> {
    ident: &'a Ident,
    kind: LineRefKind,
}

struct PlainField<'a> {
    ident: &'a Ident,
    ty: &'a Type,
}

enum LineRefKind {
    String,
    U128,
}
