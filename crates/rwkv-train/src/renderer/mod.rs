//! Metric renderers and metric message types for training output.

mod bar;

pub use bar::BarMetricsRenderer;

/// Canonical display name for loss metrics.
pub const METRIC_NAME_LOSS: &str = "Loss";
/// Canonical display name for learning-rate metrics.
pub const METRIC_NAME_LEARNING_RATE: &str = "Learning Rate";

/// Training metric payload used by renderer integrations.
pub struct TrainMetricMessage {
    /// Mini-epoch index associated with the metric.
    pub mini_epoch: usize,
    /// Step index within the current mini-epoch.
    pub step_in_epoch: usize,
    /// Current loss value.
    pub loss: f64,
    /// Current learning rate, when it is available.
    pub learning_rate: Option<f64>,
}
