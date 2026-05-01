use std::{borrow::Cow, num::NonZeroUsize, sync::Arc};

use dashmap::DashSet;
use rayon::prelude::*;
use xxhash_rust::xxh3::xxh3_128;

use crate::processor::{Step, StepOutcome, file::Writer};

/// Excludes exact duplicate samples by hashing the whole text.
///
/// The first occurrence of a hash is kept. Later occurrences are excluded and
/// sent to the configured exclusion writer.
pub struct ExactHashSampleDedup {
    writer: Arc<dyn Writer + Send + Sync + 'static>,
    seen_hashes: DashSet<u128>,
}

impl ExactHashSampleDedup {
    /// Creates a whole-sample deduplicator with an empty seen-hash set.
    pub fn new(writer: Arc<dyn Writer + Send + Sync + 'static>) -> Self {
        Self {
            writer,
            seen_hashes: DashSet::new(),
        }
    }
}

impl Step for ExactHashSampleDedup {
    fn name(&self) -> &'static str {
        "ExactHashSampleDedup"
    }

    fn batch_size(&self) -> NonZeroUsize {
        NonZeroUsize::new(4096).unwrap()
    }

    fn exclusion_writer(&self) -> Arc<dyn Writer + Send + Sync + 'static> {
        Arc::clone(&self.writer)
    }

    /// Deduplicates the batch against all hashes seen by this step instance.
    fn process_batch(&self, batch: Vec<Cow<'static, str>>) -> Vec<StepOutcome> {
        let hashed: Vec<(Cow<'static, str>, u128)> = batch
            .into_par_iter()
            .map(|data| {
                let hash = xxh3_128(data.as_bytes());

                (data, hash)
            })
            .collect();

        hashed
            .into_iter()
            .map(|(data, hash)| {
                if self.seen_hashes.insert(hash) {
                    StepOutcome::Keep(data)
                } else {
                    StepOutcome::Exclude(data)
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
        let dedup = ExactHashSampleDedup::new(Arc::new(NoopWriter));
        let outcomes = dedup.process_batch(vec![
            Cow::Borrowed("same"),
            Cow::Borrowed("unique"),
            Cow::Borrowed("same"),
        ]);

        assert_eq!(outcomes.len(), 3);

        match &outcomes[0] {
            StepOutcome::Keep(data) => assert_eq!(data.as_ref(), "same"),
            StepOutcome::Exclude(_) => panic!("expected Keep"),
        }

        match &outcomes[1] {
            StepOutcome::Keep(data) => assert_eq!(data.as_ref(), "unique"),
            StepOutcome::Exclude(_) => panic!("expected Keep"),
        }

        match &outcomes[2] {
            StepOutcome::Exclude(data) => assert_eq!(data.as_ref(), "same"),
            StepOutcome::Keep(_) => panic!("expected Exclude"),
        }

        let outcomes = dedup.process_batch(vec![Cow::Borrowed("unique")]);

        match &outcomes[0] {
            StepOutcome::Exclude(data) => assert_eq!(data.as_ref(), "unique"),
            StepOutcome::Keep(_) => panic!("expected Exclude"),
        }
    }
}
