use crate::ExportError;

pub(crate) fn validate_supported_keys<'a>(
    keys: impl IntoIterator<Item = &'a str>,
) -> Result<(), ExportError> {
    for key in keys {
        let key = key.to_ascii_lowercase();
        if is_unsupported_key(&key) {
            return Err(ExportError::UnsupportedLayout(key));
        }
    }

    Ok(())
}

pub(crate) fn should_transpose_st_key(key: &str) -> bool {
    const SUFFIXES: &[&str] = &[
        ".att.w1",
        ".att.w2",
        ".att.a1",
        ".att.a2",
        ".att.g1",
        ".att.g2",
        ".att.v1",
        ".att.v2",
        ".time_state",
    ];

    SUFFIXES.iter().any(|suffix| key.ends_with(suffix)) || key.contains(".lora.0")
}

pub(crate) fn should_transpose_burn_path(path: &str) -> bool {
    const SUFFIXES: &[&str] = &[
        ".param_weight_decay_lora.w_a",
        ".param_weight_decay_lora.w_b",
        ".param_learning_rate_lora.w_a",
        ".param_learning_rate_lora.w_b",
        ".param_output_gate_lora.w_a",
        ".param_output_gate_lora.w_b",
        ".param_value_residual_lora.w_a",
        ".param_value_residual_lora.w_b",
    ];

    SUFFIXES.iter().any(|suffix| path.ends_with(suffix))
}

fn is_unsupported_key(key: &str) -> bool {
    key.contains("deepembed")
        || key.contains("deep_embed")
        || key.contains("time_maa")
        || key.contains("time_faaaa")
        || key.contains("time_decay")
        || key.contains("time_first")
        || key.starts_with("rwkv7a.")
        || key.starts_with("rwkv7b.")
}

pub(crate) const ST_TO_BURN_KEY_MAPPINGS: &[(&str, &str)] = &[
    (r"^emb\.weight$", "embed.weight"),
    (
        r"^blocks\.0\.ln0\.weight$",
        "layer_norm_for_first_cell.gamma",
    ),
    (r"^blocks\.0\.ln0\.bias$", "layer_norm_for_first_cell.beta"),
    (r"^ln_out\.weight$", "layer_norm_for_unembed.gamma"),
    (r"^ln_out\.bias$", "layer_norm_for_unembed.beta"),
    (r"^head\.weight$", "unembed.weight"),
    (
        r"^blocks\.([0-9]+)\.ln1\.weight$",
        "cells.cells.$1.pre_layer_norm_for_time_mix.gamma",
    ),
    (
        r"^blocks\.([0-9]+)\.ln1\.bias$",
        "cells.cells.$1.pre_layer_norm_for_time_mix.beta",
    ),
    (
        r"^blocks\.([0-9]+)\.ln2\.weight$",
        "cells.cells.$1.pre_layer_norm_for_channel_mix.gamma",
    ),
    (
        r"^blocks\.([0-9]+)\.ln2\.bias$",
        "cells.cells.$1.pre_layer_norm_for_channel_mix.beta",
    ),
    (
        r"^blocks\.([0-9]+)\.att\.receptance\.weight$",
        "cells.cells.$1.time_mixer.weight_prepare.projection_receptance.weight",
    ),
    (
        r"^blocks\.([0-9]+)\.att\.key\.weight$",
        "cells.cells.$1.time_mixer.weight_prepare.projection_key.weight",
    ),
    (
        r"^blocks\.([0-9]+)\.att\.value\.weight$",
        "cells.cells.$1.time_mixer.weight_prepare.projection_value.weight",
    ),
    (
        r"^blocks\.([0-9]+)\.att\.output\.weight$",
        "cells.cells.$1.time_mixer.gated_readout.projection_output.weight",
    ),
    (
        r"^blocks\.([0-9]+)\.att\.ln_x\.weight$",
        "cells.cells.$1.time_mixer.gated_readout.group_norm.gamma",
    ),
    (
        r"^blocks\.([0-9]+)\.att\.ln_x\.bias$",
        "cells.cells.$1.time_mixer.gated_readout.group_norm.beta",
    ),
    (
        r"^blocks\.([0-9]+)\.att\.k_k$",
        "cells.cells.$1.time_mixer.weight_prepare.param_key_removal",
    ),
    (
        r"^blocks\.([0-9]+)\.att\.r_k$",
        "cells.cells.$1.time_mixer.gated_readout.param_receptance_key_bonus",
    ),
    (
        r"^blocks\.([0-9]+)\.att\.k_a$",
        "cells.cells.$1.time_mixer.weight_prepare.param_key_replacement",
    ),
    (
        r"^blocks\.([0-9]+)\.att\.w0$",
        "cells.cells.$1.time_mixer.weight_prepare.param_weight_decay_lora.bias",
    ),
    (
        r"^blocks\.([0-9]+)\.att\.w1$",
        "cells.cells.$1.time_mixer.weight_prepare.param_weight_decay_lora.w_a",
    ),
    (
        r"^blocks\.([0-9]+)\.att\.w2$",
        "cells.cells.$1.time_mixer.weight_prepare.param_weight_decay_lora.w_b",
    ),
    (
        r"^blocks\.([0-9]+)\.att\.a0$",
        "cells.cells.$1.time_mixer.weight_prepare.param_learning_rate_lora.bias",
    ),
    (
        r"^blocks\.([0-9]+)\.att\.a1$",
        "cells.cells.$1.time_mixer.weight_prepare.param_learning_rate_lora.w_a",
    ),
    (
        r"^blocks\.([0-9]+)\.att\.a2$",
        "cells.cells.$1.time_mixer.weight_prepare.param_learning_rate_lora.w_b",
    ),
    (
        r"^blocks\.([0-9]+)\.att\.g1$",
        "cells.cells.$1.time_mixer.gated_readout.param_output_gate_lora.w_a",
    ),
    (
        r"^blocks\.([0-9]+)\.att\.g2$",
        "cells.cells.$1.time_mixer.gated_readout.param_output_gate_lora.w_b",
    ),
    (
        r"^blocks\.([0-9]+)\.att\.v0$",
        "cells.cells.$1.time_mixer.weight_prepare.param_value_residual_lora.bias",
    ),
    (
        r"^blocks\.([0-9]+)\.att\.v1$",
        "cells.cells.$1.time_mixer.weight_prepare.param_value_residual_lora.w_a",
    ),
    (
        r"^blocks\.([0-9]+)\.att\.v2$",
        "cells.cells.$1.time_mixer.weight_prepare.param_value_residual_lora.w_b",
    ),
    (
        r"^blocks\.([0-9]+)\.att\.x_r$",
        "cells.cells.$1.time_mixer.weight_prepare.param_receptance",
    ),
    (
        r"^blocks\.([0-9]+)\.att\.x_w$",
        "cells.cells.$1.time_mixer.weight_prepare.param_weight_decay",
    ),
    (
        r"^blocks\.([0-9]+)\.att\.x_k$",
        "cells.cells.$1.time_mixer.weight_prepare.param_key",
    ),
    (
        r"^blocks\.([0-9]+)\.att\.x_v$",
        "cells.cells.$1.time_mixer.weight_prepare.param_value",
    ),
    (
        r"^blocks\.([0-9]+)\.att\.x_a$",
        "cells.cells.$1.time_mixer.weight_prepare.param_learning_rate",
    ),
    (
        r"^blocks\.([0-9]+)\.att\.x_g$",
        "cells.cells.$1.time_mixer.gated_readout.param_gate",
    ),
    (
        r"^blocks\.([0-9]+)\.ffn\.key\.weight$",
        "cells.cells.$1.channel_mixer.key.weight",
    ),
    (
        r"^blocks\.([0-9]+)\.ffn\.value\.weight$",
        "cells.cells.$1.channel_mixer.value.weight",
    ),
    (
        r"^blocks\.([0-9]+)\.ffn\.x_k$",
        "cells.cells.$1.channel_mixer.token_shift_diff_scale",
    ),
];

pub(crate) const BURN_TO_ST_KEY_MAPPINGS: &[(&str, &str)] = &[
    (r"^embed\.weight$", "emb.weight"),
    (r"^layer_norm_for_first_cell\.gamma$", "blocks.0.ln0.weight"),
    (r"^layer_norm_for_first_cell\.beta$", "blocks.0.ln0.bias"),
    (r"^layer_norm_for_unembed\.gamma$", "ln_out.weight"),
    (r"^layer_norm_for_unembed\.beta$", "ln_out.bias"),
    (r"^unembed\.weight$", "head.weight"),
    (
        r"^cells\.cells\.([0-9]+)\.pre_layer_norm_for_time_mix\.gamma$",
        "blocks.$1.ln1.weight",
    ),
    (
        r"^cells\.cells\.([0-9]+)\.pre_layer_norm_for_time_mix\.beta$",
        "blocks.$1.ln1.bias",
    ),
    (
        r"^cells\.cells\.([0-9]+)\.pre_layer_norm_for_channel_mix\.gamma$",
        "blocks.$1.ln2.weight",
    ),
    (
        r"^cells\.cells\.([0-9]+)\.pre_layer_norm_for_channel_mix\.beta$",
        "blocks.$1.ln2.bias",
    ),
    (
        r"^cells\.cells\.([0-9]+)\.time_mixer\.weight_prepare\.projection_receptance\.weight$",
        "blocks.$1.att.receptance.weight",
    ),
    (
        r"^cells\.cells\.([0-9]+)\.time_mixer\.weight_prepare\.projection_key\.weight$",
        "blocks.$1.att.key.weight",
    ),
    (
        r"^cells\.cells\.([0-9]+)\.time_mixer\.weight_prepare\.projection_value\.weight$",
        "blocks.$1.att.value.weight",
    ),
    (
        r"^cells\.cells\.([0-9]+)\.time_mixer\.gated_readout\.projection_output\.weight$",
        "blocks.$1.att.output.weight",
    ),
    (
        r"^cells\.cells\.([0-9]+)\.time_mixer\.gated_readout\.group_norm\.gamma$",
        "blocks.$1.att.ln_x.weight",
    ),
    (
        r"^cells\.cells\.([0-9]+)\.time_mixer\.gated_readout\.group_norm\.beta$",
        "blocks.$1.att.ln_x.bias",
    ),
    (
        r"^cells\.cells\.([0-9]+)\.time_mixer\.weight_prepare\.param_key_removal$",
        "blocks.$1.att.k_k",
    ),
    (
        r"^cells\.cells\.([0-9]+)\.time_mixer\.gated_readout\.param_receptance_key_bonus$",
        "blocks.$1.att.r_k",
    ),
    (
        r"^cells\.cells\.([0-9]+)\.time_mixer\.weight_prepare\.param_key_replacement$",
        "blocks.$1.att.k_a",
    ),
    (
        r"^cells\.cells\.([0-9]+)\.time_mixer\.weight_prepare\.param_weight_decay_lora\.bias$",
        "blocks.$1.att.w0",
    ),
    (
        r"^cells\.cells\.([0-9]+)\.time_mixer\.weight_prepare\.param_weight_decay_lora\.w_a$",
        "blocks.$1.att.w1",
    ),
    (
        r"^cells\.cells\.([0-9]+)\.time_mixer\.weight_prepare\.param_weight_decay_lora\.w_b$",
        "blocks.$1.att.w2",
    ),
    (
        r"^cells\.cells\.([0-9]+)\.time_mixer\.weight_prepare\.param_learning_rate_lora\.bias$",
        "blocks.$1.att.a0",
    ),
    (
        r"^cells\.cells\.([0-9]+)\.time_mixer\.weight_prepare\.param_learning_rate_lora\.w_a$",
        "blocks.$1.att.a1",
    ),
    (
        r"^cells\.cells\.([0-9]+)\.time_mixer\.weight_prepare\.param_learning_rate_lora\.w_b$",
        "blocks.$1.att.a2",
    ),
    (
        r"^cells\.cells\.([0-9]+)\.time_mixer\.gated_readout\.param_output_gate_lora\.w_a$",
        "blocks.$1.att.g1",
    ),
    (
        r"^cells\.cells\.([0-9]+)\.time_mixer\.gated_readout\.param_output_gate_lora\.w_b$",
        "blocks.$1.att.g2",
    ),
    (
        r"^cells\.cells\.([0-9]+)\.time_mixer\.weight_prepare\.param_value_residual_lora\.bias$",
        "blocks.$1.att.v0",
    ),
    (
        r"^cells\.cells\.([0-9]+)\.time_mixer\.weight_prepare\.param_value_residual_lora\.w_a$",
        "blocks.$1.att.v1",
    ),
    (
        r"^cells\.cells\.([0-9]+)\.time_mixer\.weight_prepare\.param_value_residual_lora\.w_b$",
        "blocks.$1.att.v2",
    ),
    (
        r"^cells\.cells\.([0-9]+)\.time_mixer\.weight_prepare\.param_receptance$",
        "blocks.$1.att.x_r",
    ),
    (
        r"^cells\.cells\.([0-9]+)\.time_mixer\.weight_prepare\.param_weight_decay$",
        "blocks.$1.att.x_w",
    ),
    (
        r"^cells\.cells\.([0-9]+)\.time_mixer\.weight_prepare\.param_key$",
        "blocks.$1.att.x_k",
    ),
    (
        r"^cells\.cells\.([0-9]+)\.time_mixer\.weight_prepare\.param_value$",
        "blocks.$1.att.x_v",
    ),
    (
        r"^cells\.cells\.([0-9]+)\.time_mixer\.weight_prepare\.param_learning_rate$",
        "blocks.$1.att.x_a",
    ),
    (
        r"^cells\.cells\.([0-9]+)\.time_mixer\.gated_readout\.param_gate$",
        "blocks.$1.att.x_g",
    ),
    (
        r"^cells\.cells\.([0-9]+)\.channel_mixer\.key\.weight$",
        "blocks.$1.ffn.key.weight",
    ),
    (
        r"^cells\.cells\.([0-9]+)\.channel_mixer\.value\.weight$",
        "blocks.$1.ffn.value.weight",
    ),
    (
        r"^cells\.cells\.([0-9]+)\.channel_mixer\.token_shift_diff_scale$",
        "blocks.$1.ffn.x_k",
    ),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_mapping() {
        assert!(ST_TO_BURN_KEY_MAPPINGS.contains(&(r"^emb\.weight$", "embed.weight")));
        assert!(ST_TO_BURN_KEY_MAPPINGS.contains(&(
            r"^blocks\.([0-9]+)\.ffn\.key\.weight$",
            "cells.cells.$1.channel_mixer.key.weight"
        )));
        assert!(BURN_TO_ST_KEY_MAPPINGS.contains(&(
            r"^cells\.cells\.([0-9]+)\.time_mixer\.weight_prepare\.param_weight_decay_lora\.w_a$",
            "blocks.$1.att.w1"
        )));
    }

    #[test]
    fn validate_supported_keys() {
        assert!(super::validate_supported_keys(["emb.weight", "blocks.0.att.r_k"]).is_ok());
        assert!(super::validate_supported_keys(["blocks.0.att.time_maa_x"]).is_err());
        assert!(super::validate_supported_keys(["deep_embed.weight"]).is_err());
    }

    #[test]
    fn should_transpose_st_key() {
        assert!(super::should_transpose_st_key("blocks.0.att.w1"));
        assert!(super::should_transpose_st_key("blocks.0.att.time_state"));
        assert!(super::should_transpose_st_key("blocks.0.att.lora.0.weight"));
        assert!(!super::should_transpose_st_key("blocks.0.att.r_k"));
    }

    #[test]
    fn should_transpose_burn_path() {
        assert!(super::should_transpose_burn_path(
            "cells.cells.0.time_mixer.weight_prepare.param_weight_decay_lora.w_a"
        ));
        assert!(!super::should_transpose_burn_path(
            "cells.cells.0.time_mixer.weight_prepare.param_weight_decay_lora.bias"
        ));
    }
}
