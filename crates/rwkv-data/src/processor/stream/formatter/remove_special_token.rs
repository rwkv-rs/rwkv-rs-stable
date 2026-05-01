use std::{borrow::Cow, num::NonZeroUsize, sync::Arc};

use once_cell::sync::Lazy;
use rayon::prelude::*;
use regex::Regex;

use crate::processor::{Step, StepOutcome, file::Writer};

static BOS_TOKEN_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"<BOS_TOKEN>").unwrap());

/// Removes literal `<BOS_TOKEN>` markers from text.
///
/// Every input item is kept after replacement.
pub struct RemoveSpecialTokenFormatter {
    writer: Arc<dyn Writer + Send + Sync + 'static>,
}

impl RemoveSpecialTokenFormatter {
    /// Creates a special-token formatter.
    pub fn new(writer: Arc<dyn Writer + Send + Sync + 'static>) -> Self {
        Self { writer }
    }
}

impl Step for RemoveSpecialTokenFormatter {
    fn name(&self) -> &'static str {
        "RemoveSpecialToken"
    }

    fn batch_size(&self) -> NonZeroUsize {
        NonZeroUsize::new(4096).unwrap()
    }

    fn exclusion_writer(&self) -> Arc<dyn Writer + Send + Sync + 'static> {
        Arc::clone(&self.writer)
    }

    /// Removes all literal `<BOS_TOKEN>` occurrences.
    ///
    /// # Panics
    ///
    /// Panics if the built-in special-token regular expression cannot be
    /// compiled.
    fn process_batch(&self, batch: Vec<Cow<'static, str>>) -> Vec<StepOutcome> {
        batch
            .into_par_iter()
            .map(|data| {
                let cleaned = BOS_TOKEN_REGEX.replace_all(data.as_ref(), "").into_owned();

                StepOutcome::Keep(Cow::Owned(cleaned))
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
        let formatter = RemoveSpecialTokenFormatter::new(Arc::new(NoopWriter));
        let outcomes = formatter.process_batch(vec![
            Cow::Borrowed("<BOS_TOKEN>Hello<BOS_TOKEN> world"),
            Cow::Borrowed("plain text"),
        ]);

        assert_eq!(outcomes.len(), 2);

        match &outcomes[0] {
            StepOutcome::Keep(data) => assert_eq!(data.as_ref(), "Hello world"),
            StepOutcome::Exclude(_) => panic!("expected Keep"),
        }

        match &outcomes[1] {
            StepOutcome::Keep(data) => assert_eq!(data.as_ref(), "plain text"),
            StepOutcome::Exclude(_) => panic!("expected Keep"),
        }
    }
}
