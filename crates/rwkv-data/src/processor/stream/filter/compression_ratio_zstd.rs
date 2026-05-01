use std::{borrow::Cow, num::NonZeroUsize, sync::Arc};

use rayon::prelude::*;

use crate::processor::{Step, StepOutcome, file::Writer};

/// Filters text by zstd compressed-size ratio.
///
/// Empty text is excluded. Text shorter than 256 bytes bypasses the compression
/// check and is kept. Longer text is kept when compressed bytes divided by
/// original bytes is in `0.18..=0.92`.
pub struct ZstdCompressionFilterStep {
    writer: Arc<dyn Writer + Send + Sync + 'static>,
}

impl ZstdCompressionFilterStep {
    /// Creates a zstd compression-ratio filter.
    pub fn new(writer: Arc<dyn Writer + Send + Sync + 'static>) -> Self {
        Self { writer }
    }

    fn should_filter(&self, text: &str) -> bool {
        const MIN_RATIO: f64 = 0.18; // Low compression ratio -> excessive repetition
        const MAX_RATIO: f64 = 0.92; // High compression ratio -> near random/hash
        const LEVEL: i32 = 3; // Light compression level
        const MIN_TEXT_BYTES: usize = 256; // Skip compression check for too short text
        if text.is_empty() {
            return true;
        }

        let payload = text.as_bytes();

        let original_size = payload.len();

        if original_size < MIN_TEXT_BYTES {
            return false;
        }

        let compressed = zstd::encode_all(payload, LEVEL).unwrap();

        let ratio = compressed.len() as f64 / original_size as f64;

        !(MIN_RATIO..=MAX_RATIO).contains(&ratio)
    }
}

impl Step for ZstdCompressionFilterStep {
    fn name(&self) -> &'static str {
        "ZstdCompressionFilter"
    }

    fn batch_size(&self) -> NonZeroUsize {
        NonZeroUsize::new(4096).unwrap()
    }

    fn exclusion_writer(&self) -> Arc<dyn Writer + Send + Sync + 'static> {
        Arc::clone(&self.writer)
    }

    /// Applies the zstd ratio threshold to each item.
    ///
    /// # Panics
    ///
    /// Panics if zstd compression fails.
    fn process_batch(&self, batch: Vec<Cow<'static, str>>) -> Vec<StepOutcome> {
        batch
            .into_par_iter()
            .map(|data| {
                if self.should_filter(data.as_ref()) {
                    StepOutcome::Exclude(data)
                } else {
                    StepOutcome::Keep(data)
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::{future::Future, pin::Pin};

    use tokio::sync::mpsc::Receiver;

    use super::*;
    use crate::processor::file::DataItem;

    struct NoopWriter;

    impl Writer for NoopWriter {
        fn run(&self, _rx: Receiver<DataItem>) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
            Box::pin(async {})
        }
    }

    #[test]
    fn process_batch() {
        let filter = ZstdCompressionFilterStep::new(Arc::new(NoopWriter));
        let repeated = "a".repeat(1024);
        let outcomes = filter.process_batch(vec![
            Cow::Borrowed(""),
            Cow::Borrowed("short text"),
            Cow::Owned(repeated.clone()),
        ]);

        assert_eq!(outcomes.len(), 3);

        match &outcomes[0] {
            StepOutcome::Exclude(data) => assert_eq!(data.as_ref(), ""),
            StepOutcome::Keep(_) => panic!("expected Exclude"),
        }

        match &outcomes[1] {
            StepOutcome::Keep(data) => assert_eq!(data.as_ref(), "short text"),
            StepOutcome::Exclude(_) => panic!("expected Keep"),
        }

        match &outcomes[2] {
            StepOutcome::Exclude(data) => assert_eq!(data.as_ref(), repeated),
            StepOutcome::Keep(_) => panic!("expected Exclude"),
        }
    }
}
