use syn::{
    Attribute,
    Data,
    DeriveInput,
    Fields,
    FieldsNamed,
    GenericArgument,
    PathArguments,
    Type,
};

pub(crate) fn named_struct_fields<'a>(
    input: &'a DeriveInput,
    derive_name: &str,
) -> syn::Result<(&'a syn::Ident, &'a FieldsNamed)> {
    let Data::Struct(data_struct) = &input.data else {
        return Err(syn::Error::new_spanned(
            input,
            format!("{derive_name} can only be derived for structs"),
        ));
    };

    let Fields::Named(fields) = &data_struct.fields else {
        return Err(syn::Error::new_spanned(
            &data_struct.fields,
            format!("{derive_name} can only be derived for structs with named fields"),
        ));
    };

    Ok((&input.ident, fields))
}

pub(crate) fn has_attr(attrs: &[Attribute], name: &str) -> bool {
    attrs.iter().any(|attr| attr.path().is_ident(name))
}

pub(crate) fn option_inner_type(ty: &Type) -> Option<&Type> {
    let Type::Path(type_path) = ty else {
        return None;
    };

    let segment = type_path.path.segments.last()?;
    if segment.ident != "Option" {
        return None;
    }

    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };

    let Some(GenericArgument::Type(inner_type)) = args.args.first() else {
        return None;
    };

    Some(inner_type)
}
