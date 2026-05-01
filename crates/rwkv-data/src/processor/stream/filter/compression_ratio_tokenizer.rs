use std::{
    borrow::Cow,
    num::NonZeroUsize,
    sync::{Arc, Mutex},
};

use rayon::prelude::*;

use crate::{
    processor::{Step, StepOutcome, file::Writer},
    tokenizer::Tokenizer,
};

/// Filters text by tokenizer tokens-per-character ratio.
///
/// Empty text is excluded. Text shorter than 50 Unicode scalar values bypasses
/// the ratio check and is kept. Longer text is kept when token count divided by
/// character count is in `0.08..=0.85`.
pub struct TokenizerCompressionFilterStep {
    writer: Arc<dyn Writer + Send + Sync + 'static>,
    vocab_path: String,
    tokenizer: Mutex<Option<Arc<Tokenizer>>>,
}

impl TokenizerCompressionFilterStep {
    /// Creates a tokenizer compression-ratio filter.
    ///
    /// Call [`set_vocab_path`](Self::set_vocab_path) before running the step.
    pub fn new(writer: Arc<dyn Writer + Send + Sync + 'static>) -> Self {
        Self {
            writer,
            vocab_path: String::new(),
            tokenizer: Mutex::new(None),
        }
    }

    /// Sets the tokenizer vocabulary path and clears the cached tokenizer.
    pub fn set_vocab_path(&mut self, vocab_path: String) {
        self.vocab_path = vocab_path;

        if let Ok(mut guard) = self.tokenizer.lock() {
            *guard = None;
        }
    }

    fn tokenizer(&self) -> Arc<Tokenizer> {
        let vocab_path = self.vocab_path.clone();

        let mut guard = self.tokenizer.lock().unwrap();

        guard
            .get_or_insert_with(|| {
                assert!(
                    !vocab_path.is_empty(),
                    "Tokenizer vocab path must be configured before running the pipeline"
                );

                Arc::new(Tokenizer::new(&vocab_path).expect("failed to load tokenizer vocab"))
            })
            .clone()
    }

    fn should_filter(&self, text: &str, tokenizer: &Tokenizer) -> bool {
        const MIN_RATIO: f64 = 0.08; // Low compression ratio -> excessive repetition or very long tokens
        const MAX_RATIO: f64 = 0.85; // High compression ratio -> too many special chars or poor tokenization
        const MIN_TEXT_LENGTH: usize = 50; // Skip compression check for too short text
        if text.is_empty() {
            return true;
        }

        let char_count = text.chars().count();

        if char_count < MIN_TEXT_LENGTH {
            return false;
        }

        let token_ids = tokenizer.encode(text, false);

        let token_count = token_ids.len();

        if token_count == 0 {
            return true;
        }

        let ratio = token_count as f64 / char_count as f64;

        !(MIN_RATIO..=MAX_RATIO).contains(&ratio)
    }
}

impl Step for TokenizerCompressionFilterStep {
    fn name(&self) -> &'static str {
        "TokenizerCompressionFilter"
    }

    fn batch_size(&self) -> NonZeroUsize {
        NonZeroUsize::new(4096).unwrap()
    }

    fn exclusion_writer(&self) -> Arc<dyn Writer + Send + Sync + 'static> {
        Arc::clone(&self.writer)
    }

    /// Applies the tokenizer ratio threshold to each item.
    ///
    /// # Panics
    ///
    /// Panics if the tokenizer mutex is poisoned, the vocabulary path has not
    /// been configured, or the tokenizer vocabulary cannot be loaded.
    fn process_batch(&self, batch: Vec<Cow<'static, str>>) -> Vec<StepOutcome> {
        let tokenizer = self.tokenizer();

        batch
            .into_par_iter()
            .map(|data| {
                if self.should_filter(data.as_ref(), &tokenizer) {
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
    use std::{future::Future, path::PathBuf, pin::Pin};

    use tokio::sync::mpsc::Receiver;

    use super::*;
    use crate::processor::file::DataItem;

    struct NoopWriter;

    impl Writer for NoopWriter {
        fn run(&self, _rx: Receiver<DataItem>) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
            Box::pin(async {})
        }
    }

    fn vocab_path() -> String {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../vocab/rwkv_vocab_v20230424.txt")
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn process_batch() {
        let mut filter = TokenizerCompressionFilterStep::new(Arc::new(NoopWriter));
        let normal_long = "The quick brown fox jumps over the lazy dog. ".repeat(4);
        let emoji_heavy = "😀".repeat(64);

        filter.set_vocab_path(vocab_path());

        let outcomes = filter.process_batch(vec![
            Cow::Borrowed(""),
            Cow::Borrowed("short text"),
            Cow::Owned(normal_long.clone()),
            Cow::Owned(emoji_heavy.clone()),
        ]);

        assert_eq!(outcomes.len(), 4);

        match &outcomes[0] {
            StepOutcome::Exclude(data) => assert_eq!(data.as_ref(), ""),
            StepOutcome::Keep(_) => panic!("expected Exclude"),
        }

        match &outcomes[1] {
            StepOutcome::Keep(data) => assert_eq!(data.as_ref(), "short text"),
            StepOutcome::Exclude(_) => panic!("expected Keep"),
        }

        match &outcomes[2] {
            StepOutcome::Keep(data) => assert_eq!(data.as_ref(), normal_long),
            StepOutcome::Exclude(_) => panic!("expected Keep"),
        }

        match &outcomes[3] {
            StepOutcome::Exclude(data) => assert_eq!(data.as_ref(), emoji_heavy),
            StepOutcome::Keep(_) => panic!("expected Exclude"),
        }
    }
}
