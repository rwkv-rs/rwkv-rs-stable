use std::{
    collections::{BTreeMap, HashMap},
    fs,
    io,
    path::Path,
};

use burn::{prelude::Backend, tensor::Element};
use burn_store::{ModuleSnapshot, ModuleStore, PytorchStore, SafetensorsStore};
use rwkv_nn::models::lm::{RwkvLM, RwkvLMConfig};
use safetensors::{SafeTensors, tensor::serialize_to_file};

use crate::{
    ConvertOptions,
    ExportError,
    RwkvLmStInfo,
    keys::{BURN_TO_ST_KEY_MAPPINGS, ST_TO_BURN_KEY_MAPPINGS, validate_supported_keys},
    tensor::{AdapterDirection, RwkvLmStAdapter, materialize_st_tensor},
};

/// Converts a local RWKV7 G1 `.pth` checkpoint into a web-rwkv `.safetensors` file.
pub fn rwkv_lm_pth2st(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    options: ConvertOptions,
) -> Result<(), ExportError> {
    let input = input.as_ref();
    let output = output.as_ref();
    ensure_output_can_be_written(output, options)?;

    let mut store = PytorchStore::from_file(input).map_indices_contiguous(false);
    let snapshots = store
        .get_all_snapshots()
        .map_err(|error| ExportError::Pytorch(error.to_string()))?;
    validate_supported_keys(snapshots.keys().map(String::as_str))?;

    let mut tensors = BTreeMap::new();
    for (name, snapshot) in snapshots {
        let name = name.to_ascii_lowercase();
        let tensor = materialize_st_tensor(&name, snapshot)?;
        tensors.insert(name, tensor);
    }

    let metadata = HashMap::from([("format".to_string(), "pt".to_string())]);
    serialize_to_file(tensors, Some(metadata), output)?;

    Ok(())
}

/// Validates a RWKV LM `.safetensors` file and returns inferred RWKV7 G1 shape information.
pub fn check_st(path: impl AsRef<Path>) -> Result<RwkvLmStInfo, ExportError> {
    let bytes = fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    validate_supported_keys(tensors.names().iter().copied())?;
    infer_st_info(&tensors)
}

/// Loads a RWKV LM `.safetensors` file into the repository's Burn [`RwkvLM`] model.
pub fn rwkv_lm_load_st<B: Backend>(
    path: impl AsRef<Path>,
    device: &B::Device,
) -> Result<RwkvLM<B>, ExportError> {
    let info = check_st(path.as_ref())?;
    let config = RwkvLMConfig::new(
        info.num_cells,
        info.vocab_size,
        info.embedded_dim,
        info.num_heads,
        info.head_size,
    );
    let mut model = config.init(device);
    let mut store = SafetensorsStore::from_file(path.as_ref())
        .with_from_adapter(
            RwkvLmStAdapter::new(AdapterDirection::StToBurn).with_load_dtype(B::FloatElem::dtype()),
        )
        .allow_partial(true);
    for (st, burn) in ST_TO_BURN_KEY_MAPPINGS {
        store = store.with_key_remapping(st, *burn);
    }
    let result = model
        .load_from(&mut store)
        .map_err(|error| ExportError::BurnSafetensors(error.to_string()))?;

    if !result.errors.is_empty() {
        return Err(ExportError::InvalidTensorSet(format!(
            "failed to apply tensors:\n{}",
            result
        )));
    }

    let missing_non_state = result
        .missing
        .iter()
        .map(|(path, _)| path.as_str())
        .filter(|path| !path.starts_with("state."))
        .collect::<Vec<_>>();
    if !missing_non_state.is_empty() {
        return Err(ExportError::InvalidTensorSet(format!(
            "missing model tensors: {}",
            missing_non_state.join(", ")
        )));
    }

    Ok(model)
}

/// Saves a Burn [`RwkvLM`] model into a web-rwkv `.safetensors` file.
pub fn rwkv_lm_save_st<B: Backend>(
    model: &RwkvLM<B>,
    output: impl AsRef<Path>,
    options: ConvertOptions,
) -> Result<(), ExportError> {
    let output = output.as_ref();
    ensure_output_can_be_written(output, options)?;

    let mut store = SafetensorsStore::from_file(output)
        .clear_metadata()
        .metadata("format", "pt")
        .with_to_adapter(RwkvLmStAdapter::new(AdapterDirection::BurnToSt))
        .with_predicate(|path, _| !path.starts_with("state."))
        .overwrite(options.overwrite);
    for (burn, st) in BURN_TO_ST_KEY_MAPPINGS {
        store = store.with_key_remapping(burn, *st);
    }
    model
        .save_into(&mut store)
        .map_err(|error| ExportError::BurnSafetensors(error.to_string()))?;

    Ok(())
}

fn ensure_output_can_be_written(path: &Path, options: ConvertOptions) -> Result<(), ExportError> {
    if path.exists() && !options.overwrite {
        return Err(ExportError::Io(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("File already exists: {}", path.display()),
        )));
    }

    Ok(())
}

fn infer_st_info(tensors: &SafeTensors<'_>) -> Result<RwkvLmStInfo, ExportError> {
    let names = tensors.names();
    let emb = tensors
        .tensor("emb.weight")
        .map_err(|_| ExportError::InvalidTensorSet("missing emb.weight".to_string()))?;
    let emb_shape = emb.shape();
    if emb_shape.len() != 2 {
        return Err(ExportError::InvalidTensorSet(format!(
            "emb.weight must be rank 2, got {:?}",
            emb_shape
        )));
    }

    let mut max_block = None;
    for name in &names {
        let Some(suffix) = name.strip_prefix("blocks.") else {
            continue;
        };
        let Some((index, _)) = suffix.split_once('.') else {
            continue;
        };
        let Ok(block) = index.parse::<usize>() else {
            continue;
        };
        max_block = Some(max_block.map_or(block, |max: usize| max.max(block)));
    }
    let num_cells = max_block
        .ok_or_else(|| ExportError::InvalidTensorSet("missing blocks.* keys".to_string()))?
        + 1;

    let r_k_name = "blocks.0.att.r_k";
    let r_k = tensors
        .tensor(r_k_name)
        .map_err(|_| ExportError::InvalidTensorSet(format!("missing {r_k_name}")))?;
    let r_k_shape = r_k.shape();
    if r_k_shape.len() != 2 {
        return Err(ExportError::InvalidTensorSet(format!(
            "{r_k_name} must be rank 2, got {:?}",
            r_k_shape
        )));
    }

    let info = RwkvLmStInfo {
        num_tensors: names.len(),
        num_cells,
        vocab_size: emb_shape[0],
        embedded_dim: emb_shape[1],
        num_heads: r_k_shape[0],
        head_size: r_k_shape[1],
    };

    if info.num_heads * info.head_size != info.embedded_dim {
        return Err(ExportError::InvalidTensorSet(format!(
            "blocks.0.att.r_k shape {:?} is incompatible with embedding dim {}",
            r_k_shape, info.embedded_dim
        )));
    }

    Ok(info)
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, path::Path};

    use burn::{backend::Cpu, tensor::Device};
    use half::f16;
    use safetensors::{
        SafeTensors,
        tensor::{Dtype as SafeDtype, TensorView},
    };
    use tempfile::tempdir;

    use super::*;

    type TestBackend = Cpu<f32, i32>;

    #[test]
    fn check_st() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        write_minimal_st(&path);

        let info = super::check_st(&path).unwrap();

        assert_eq!(
            info,
            RwkvLmStInfo {
                num_tensors: 3,
                num_cells: 2,
                vocab_size: 8,
                embedded_dim: 32,
                num_heads: 4,
                head_size: 8,
            }
        );
    }

    #[test]
    fn rwkv_lm_save_st_and_load_st() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        let device = Device::<TestBackend>::default();
        let mut model = RwkvLMConfig::new(2, 8, 32, 4, 8).init::<TestBackend>(&device);
        model.init_weights(&device);

        super::rwkv_lm_save_st(&model, &path, ConvertOptions { overwrite: false }).unwrap();

        let bytes = fs::read(&path).unwrap();
        let tensors = SafeTensors::deserialize(&bytes).unwrap();
        assert!(tensors.tensor("state.state").is_err());
        assert_eq!(
            tensors.tensor("emb.weight").unwrap().dtype(),
            SafeDtype::F16
        );
        assert!(tensors.tensor("blocks.0.att.w1").is_ok());

        let loaded = super::rwkv_lm_load_st::<TestBackend>(&path, &device).unwrap();
        let loaded_snapshots = loaded.collect(None, None, false);
        assert!(
            loaded_snapshots
                .iter()
                .any(|snapshot| snapshot.full_path() == "embed.weight")
        );
    }

    fn write_minimal_st(path: &Path) {
        let emb = f16_bytes(8 * 32);
        let r_k = f16_bytes(4 * 8);
        let block_1 = f16_bytes(4 * 8);
        let tensors = BTreeMap::from([
            (
                "emb.weight".to_string(),
                TensorView::new(SafeDtype::F16, vec![8, 32], &emb).unwrap(),
            ),
            (
                "blocks.0.att.r_k".to_string(),
                TensorView::new(SafeDtype::F16, vec![4, 8], &r_k).unwrap(),
            ),
            (
                "blocks.1.att.r_k".to_string(),
                TensorView::new(SafeDtype::F16, vec![4, 8], &block_1).unwrap(),
            ),
        ]);
        serialize_to_file(tensors, None, path).unwrap();
    }

    fn f16_bytes(num_values: usize) -> Vec<u8> {
        (0..num_values)
            .flat_map(|value| f16::from_f32(value as f32).to_le_bytes())
            .collect()
    }
}
