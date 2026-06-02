//! Learner initialization helpers for configuration, devices, logging, and
//! metric rendering.

use std::{
    any::Any,
    path::{Path, PathBuf},
};
#[cfg(feature = "wgpu")]
use std::any::TypeId;

use burn::{
    backend::{Autodiff, autodiff::checkpoint::strategy::CheckpointStrategy},
    prelude::Backend,
};
#[cfg(feature = "wgpu")]
use burn::backend::wgpu::WgpuDevice;
use burn_cubecl::{
    CubeBackend,
    CubeRuntime,
    cubecl::device::{Device as CubeDevice, DeviceId},
};
#[cfg(feature = "fusion")]
use burn_fusion::{Fusion, FusionBackend};
#[cfg(feature = "tui")]
use burn::train::renderer::tui::TuiMetricsRendererWrapper;
use burn_train::{
    Interrupter,
    logger::{AsyncLogger, FileLogger, Logger},
    renderer::MetricsRenderer,
};
use chrono::Local;
use log::{info, warn};
use rwkv_config::{
    load_toml,
    raw::{model::RawModelConfig, train::RawTrainConfig},
    validated::{
        model::FinalModelConfigBuilder,
        train::{FinalTrainConfigBuilder, TRAIN_CFG},
    },
};
use wandb::LogData;

use crate::{
    logger::wandb::{WandbLogger, WandbLoggerConfig, init_logger, init_metric_logger},
    renderer::BarMetricsRenderer,
    utils::{auto_create_directory, read_record_file},
};

fn resolve_path(base_dir: &Path, path: &str) -> String {
    let p = Path::new(path);
    if p.is_absolute() {
        path.to_string()
    } else {
        base_dir.join(p).to_string_lossy().to_string()
    }
}

fn resolve_model_cfg_path(config_dir: &Path, train_cfg_dir: &Path, model_cfg: &str) -> PathBuf {
    if model_cfg.contains('/') || model_cfg.contains('\\') {
        let p = PathBuf::from(model_cfg);
        if p.is_absolute() {
            p
        } else {
            train_cfg_dir.join(p)
        }
    } else {
        config_dir.join("model").join(format!("{model_cfg}.toml"))
    }
}

/// Loads a named train config and its referenced model config from a config root.
///
/// The train config is read from `config_dir/train/{train_cfg_name}.toml`.
/// Plain model config names are resolved as `config_dir/model/{name}.toml`;
/// path-like model config values are resolved relative to the train config
/// directory unless they are absolute paths.
///
/// # Panics
///
/// Panics if either TOML file cannot be read or deserialized.
pub fn init_cfg<P: AsRef<Path>>(
    config_dir: P,
    train_cfg_name: &str,
) -> (FinalModelConfigBuilder, FinalTrainConfigBuilder) {
    let config_dir = config_dir.as_ref();
    let train_cfg_path = config_dir
        .join("train")
        .join(format!("{train_cfg_name}.toml"));
    let train_cfg_dir = train_cfg_path.parent().unwrap_or_else(|| Path::new("."));

    let mut raw_train_cfg: RawTrainConfig = load_toml(&train_cfg_path);
    raw_train_cfg.fill_default();

    // Resolve relative paths against the train config directory.
    raw_train_cfg.dataset_base_path = resolve_path(train_cfg_dir, &raw_train_cfg.dataset_base_path);
    raw_train_cfg.experiment_log_base_path = raw_train_cfg
        .experiment_log_base_path
        .map(|p| resolve_path(train_cfg_dir, &p));
    raw_train_cfg.record_path = raw_train_cfg
        .record_path
        .map(|p| resolve_path(train_cfg_dir, &p));

    let model_cfg_path =
        resolve_model_cfg_path(config_dir, train_cfg_dir, raw_train_cfg.model_cfg.as_str());

    let mut raw_model_cfg: RawModelConfig = load_toml(&model_cfg_path);
    raw_model_cfg.fill_default();

    let mut model_cfg_builder = FinalModelConfigBuilder::load_from_raw(raw_model_cfg);
    let mut train_cfg_builder = FinalTrainConfigBuilder::load_from_raw(raw_train_cfg);

    model_cfg_builder.fill_auto_after_load();
    train_cfg_builder.fill_auto_after_load();

    (model_cfg_builder, train_cfg_builder)
}

/// Loads model and train configs from explicit TOML paths.
///
/// # Panics
///
/// Panics if either TOML file cannot be read or deserialized.
pub fn init_cfg_paths<P1: AsRef<Path>, P2: AsRef<Path>>(
    model_cfg_path: P1,
    train_cfg_path: P2,
) -> (FinalModelConfigBuilder, FinalTrainConfigBuilder) {
    let mut raw_model_cfg: RawModelConfig = load_toml(model_cfg_path);
    let mut raw_train_cfg: RawTrainConfig = load_toml(train_cfg_path);

    raw_model_cfg.fill_default();
    raw_train_cfg.fill_default();

    let mut model_cfg_builder = FinalModelConfigBuilder::load_from_raw(raw_model_cfg);
    let mut train_cfg_builder = FinalTrainConfigBuilder::load_from_raw(raw_train_cfg);

    model_cfg_builder.fill_auto_after_load();
    train_cfg_builder.fill_auto_after_load();

    (model_cfg_builder, train_cfg_builder)
}

/// Initializes experiment logging and updates the train builder with record state.
///
/// Returns the canonical experiment log directory. The directory is created from
/// the configured experiment log base path and experiment name. The function also
/// initializes stdout/file tracing when tracing has not already been installed,
/// reads any configured record file, and writes that record path back into the
/// train config builder.
///
/// # Panics
///
/// Panics if required builder fields are missing, the log directory cannot be
/// created or canonicalized, or the canonical log path is not valid UTF-8.
pub fn init_log(train_cfg_builder: &mut FinalTrainConfigBuilder) -> PathBuf {
    let full_experiment_log_path = auto_create_directory(
        auto_create_directory(PathBuf::from(
            train_cfg_builder.get_experiment_log_base_path().unwrap(),
        ))
        .join(train_cfg_builder.get_experiment_name().unwrap()),
    )
    .canonicalize()
    .unwrap();

    let level = train_cfg_builder.get_level().unwrap();
    #[cfg(feature = "trace")]
    let tracing_already_initialized = tracing::dispatcher::has_been_set();
    #[cfg(not(feature = "trace"))]
    let tracing_already_initialized = false;

    if !tracing_already_initialized {
        let _guard = clia_tracing_config::build()
            .filter_level(level.as_str())
            .with_ansi(true)
            .to_stdout(true)
            .directory(full_experiment_log_path.to_str().unwrap())
            .file_name("experiment.log")
            .init();
    }

    info!("log level: {}", level);

    let record_path = read_record_file(
        train_cfg_builder.get_record_path(),
        &full_experiment_log_path,
    );
    info!("Getting Record Path Completed. record_path: {record_path:?}");
    train_cfg_builder.fill_after_read_record_file(record_path);

    full_experiment_log_path
}

/// Initializes backend devices from finalized training configuration.
pub trait BackendDeviceInit: Backend {
    /// Creates and seeds the backend devices used by training.
    ///
    /// # Panics
    ///
    /// Implementations may panic if required train config fields are missing,
    /// requested device indexes cannot be represented, or the requested backend
    /// devices are unavailable.
    fn init_devices(train_cfg_builder: &FinalTrainConfigBuilder) -> Vec<Self::Device>;
}

impl<B, C> BackendDeviceInit for Autodiff<B, C>
where
    B: BackendDeviceInit,
    C: CheckpointStrategy,
{
    fn init_devices(train_cfg_builder: &FinalTrainConfigBuilder) -> Vec<Self::Device> {
        B::init_devices(train_cfg_builder)
    }
}

/// Initializes backend devices from finalized training configuration.
pub fn init_devices<B: BackendDeviceInit>(
    train_cfg_builder: &FinalTrainConfigBuilder,
) -> Vec<B::Device> {
    B::init_devices(train_cfg_builder)
}

#[cfg(feature = "fusion")]
impl<B> BackendDeviceInit for Fusion<B>
where
    B: FusionBackend + BackendDeviceInit,
{
    fn init_devices(train_cfg_builder: &FinalTrainConfigBuilder) -> Vec<Self::Device> {
        <B as BackendDeviceInit>::init_devices(train_cfg_builder)
    }
}

impl<R, F, I, BT> BackendDeviceInit for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    R::Device: CubeDevice + Any,
    F: burn_cubecl::FloatElement,
    I: burn_cubecl::IntElement,
    BT: burn_cubecl::BoolElement,
{
    fn init_devices(train_cfg_builder: &FinalTrainConfigBuilder) -> Vec<Self::Device> {
        let seed = train_cfg_builder.get_random_seed().unwrap();
        let requested = train_cfg_builder.get_num_devices_per_node().unwrap();

        #[cfg(feature = "wgpu")]
        let is_wgpu = TypeId::of::<R::Device>() == TypeId::of::<WgpuDevice>();
        #[cfg(not(feature = "wgpu"))]
        let is_wgpu = false;

        let device_ids = if is_wgpu {
            if requested <= 1 {
                vec![DeviceId::new(4, 0)]
            } else {
                let available_discrete = Self::device_count(0);
                let available_integrated = Self::device_count(1);
                let available_virtual = Self::device_count(2);
                let available_cpu = Self::device_count(3);

                let type_id = if available_discrete >= requested {
                    0
                } else if available_integrated >= requested {
                    1
                } else if available_virtual >= requested {
                    2
                } else if available_cpu >= requested {
                    3
                } else {
                    panic!(
                        "Requested {requested} WGPU devices, but only found \
                         discrete={available_discrete}, integrated={available_integrated}, \
                         virtual={available_virtual}, cpu={available_cpu}. Reduce \
                         num_devices_per_node or change backend."
                    );
                };

                (0..requested)
                    .map(|i| {
                        DeviceId::new(type_id, u16::try_from(i).expect("device index exceeds u16"))
                    })
                    .collect()
            }
        } else {
            let available = Self::device_count(0);
            let count = requested.min(available.max(1));
            (0..count)
                .map(|i| DeviceId::new(0, u16::try_from(i).expect("device index exceeds u16")))
                .collect()
        };

        let devices = device_ids
            .into_iter()
            .map(<R::Device as CubeDevice>::from_id)
            .collect::<Vec<_>>();

        for device in &devices {
            #[cfg(feature = "wgpu")]
            if let Some(wgpu_device) = (device as &dyn Any).downcast_ref::<WgpuDevice>() {
                #[cfg(feature = "metal")]
                burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Metal>(
                    wgpu_device,
                    Default::default(),
                );
                #[cfg(not(feature = "metal"))]
                burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::AutoGraphicsApi>(
                    wgpu_device,
                    Default::default(),
                );
            }

            Self::seed(device, seed);
        }

        devices
    }
}

/// Creates the asynchronous CSV-like file logger for training metrics.
///
/// The logger writes an initial
/// `global_step,epoch,step_in_epoch,learning_rate,loss` header to
/// `training_metrics.log` under `exp_log_path`.
pub fn init_file_logger(exp_log_path: &Path) -> AsyncLogger<String> {
    let mut file_logger =
        AsyncLogger::new(FileLogger::new(exp_log_path.join("training_metrics.log")));

    file_logger.log("global_step,epoch,step_in_epoch,learning_rate,loss".to_string());
    file_logger
}

/// Initializes the asynchronous Weights & Biases scalar logger when enabled.
///
/// Returns `None` when `TRAIN_CFG.upload_to_wandb` is false.
///
/// # Panics
///
/// Panics if the global train config has not been initialized, or if WandB
/// upload is enabled without the required API key or project name.
pub fn init_wandb_logger() -> Option<AsyncLogger<LogData>> {
    let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    let mut wandb_logger: Option<AsyncLogger<LogData>> = None;
    if TRAIN_CFG.get().unwrap().upload_to_wandb {
        let api_key = TRAIN_CFG.get().unwrap().wandb_api_key.as_ref().unwrap();
        let project = TRAIN_CFG
            .get()
            .unwrap()
            .wandb_project_name
            .as_ref()
            .unwrap();
        let entity = TRAIN_CFG.get().unwrap().wandb_entity_name.as_ref();

        let run_name = format!("{}_{}", TRAIN_CFG.get().unwrap().experiment_name, timestamp);
        let mut config = WandbLoggerConfig::new(api_key, project).run_name(run_name);
        if let Some(entity) = entity {
            config = config.entity(entity);
        } else {
            warn!("wandb entity name missing, falling back to default entity");
        }
        wandb_logger = Some(init_logger(config));
        info!("Wandb logger initialized.");
    }
    wandb_logger
}

/// Initializes the direct Weights & Biases metric logger when enabled.
///
/// Returns `None` when `TRAIN_CFG.upload_to_wandb` is false.
///
/// # Panics
///
/// Panics if the global train config has not been initialized, or if WandB
/// upload is enabled without the required API key or project name.
pub fn init_wandb_metric_logger() -> Option<WandbLogger> {
    let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    let mut wandb_logger: Option<WandbLogger> = None;
    if TRAIN_CFG.get().unwrap().upload_to_wandb {
        let api_key = TRAIN_CFG.get().unwrap().wandb_api_key.as_ref().unwrap();
        let project = TRAIN_CFG
            .get()
            .unwrap()
            .wandb_project_name
            .as_ref()
            .unwrap();
        let entity = TRAIN_CFG.get().unwrap().wandb_entity_name.as_ref();

        let run_name = format!("{}_{}", TRAIN_CFG.get().unwrap().experiment_name, timestamp);
        let mut config = WandbLoggerConfig::new(api_key, project).run_name(run_name);
        if let Some(entity) = entity {
            config = config.entity(entity);
        } else {
            warn!("wandb entity name missing, falling back to default entity");
        }
        wandb_logger = Some(init_metric_logger(config));
        info!("Wandb metric logger initialized.");
    }
    wandb_logger
}

/// Creates the training metrics renderer and its interrupter.
///
/// Uses the TUI renderer when `TRAIN_CFG.use_tui` is true and the `tui` feature
/// is enabled. Otherwise it falls back to the bar renderer.
///
/// # Panics
///
/// Panics if the global train config has not been initialized.
pub fn init_renderer() -> (Interrupter, Box<dyn MetricsRenderer>) {
    let interrupter = Interrupter::new();
    if TRAIN_CFG.get().unwrap().use_tui {
        #[cfg(feature = "tui")]
        {
            return (
                interrupter.clone(),
                Box::new(TuiMetricsRendererWrapper::new(interrupter, None)),
            );
        }
        #[cfg(not(feature = "tui"))]
        {
            warn!("use_tui=true but feature \"tui\" is disabled, falling back to bar renderer");
        }
    }

    (
        interrupter,
        Box::new(BarMetricsRenderer::new(
            TRAIN_CFG.get().unwrap().num_mini_epochs_auto,
        )),
    )
}
