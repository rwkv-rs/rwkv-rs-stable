use std::{borrow::Cow, collections::HashMap, num::NonZeroUsize, sync::Arc};

use rayon::prelude::*;
use unicode_script::{Script, UnicodeScript};
use unicode_segmentation::UnicodeSegmentation;

use crate::processor::{Step, StepOutcome, file::Writer};

/// Filters suspicious low-frequency Unicode script mixing.
///
/// Text with no concrete script characters is excluded. Otherwise the step
/// keeps text unless a script accounts for less than 1% of script characters,
/// appears at least three times, and mostly appears as isolated one-character
/// runs.
pub struct LanguageCharScriptFilter {
    writer: Arc<dyn Writer + Send + Sync + 'static>,
}

impl LanguageCharScriptFilter {
    /// Creates a Unicode script-mix filter.
    pub fn new(writer: Arc<dyn Writer + Send + Sync + 'static>) -> Self {
        Self { writer }
    }

    fn should_keep(&self, data: &str) -> bool {
        let graphemes = UnicodeSegmentation::graphemes(data, true).collect::<Vec<_>>();

        let mut script_seq = Vec::with_capacity(graphemes.len());

        for g in &graphemes {
            for ch in g.chars() {
                let script = ch.script();

                if script != Script::Common
                    && script != Script::Inherited
                    && script != Script::Unknown
                {
                    script_seq.push(script);

                    break;
                }
            }
        }

        if script_seq.is_empty() {
            return false;
        }

        let total = script_seq.len() as f64;

        let mut script_counts: HashMap<Script, usize> = HashMap::new();

        let mut script_runs: HashMap<Script, Vec<usize>> = HashMap::new();

        let mut i = 0;

        while i < script_seq.len() {
            let curr_script = script_seq[i];

            let mut run_len = 1;

            let mut j = i + 1;

            while j < script_seq.len() && script_seq[j] == curr_script {
                run_len += 1;

                j += 1;
            }

            *script_counts.entry(curr_script).or_default() += run_len;

            script_runs.entry(curr_script).or_default().push(run_len);

            i = j;
        }

        let has_suspicious_mix = script_counts.iter().any(|(script, &count)| {
            let ratio = count as f64 / total;

            let runs = script_runs.get(script).unwrap();

            let avg_run = count as f64 / runs.len() as f64;

            let len1_runs = runs.iter().filter(|&&len| len == 1).count();

            let ratio_len1 = len1_runs as f64 / runs.len() as f64;

            let adj_same = runs.iter().map(|&len| len.saturating_sub(1)).sum::<usize>();

            let adj_ratio = if count > 1 {
                adj_same as f64 / (count - 1) as f64
            } else {
                0.0
            };

            ratio < 0.01 && count >= 3 && avg_run <= 1.4 && ratio_len1 >= 0.7 && adj_ratio <= 0.2
        });

        !has_suspicious_mix
    }
}

impl Step for LanguageCharScriptFilter {
    fn name(&self) -> &'static str {
        "LanguageCharScriptFilter"
    }

    fn batch_size(&self) -> NonZeroUsize {
        NonZeroUsize::new(4096).unwrap()
    }

    fn exclusion_writer(&self) -> Arc<dyn Writer + Send + Sync + 'static> {
        Arc::clone(&self.writer)
    }

    /// Keeps text whose script distribution is not suspicious.
    ///
    /// # Panics
    ///
    /// Panics if internal script-run bookkeeping is inconsistent.
    fn process_batch(&self, batch: Vec<Cow<'static, str>>) -> Vec<StepOutcome> {
        batch
            .into_par_iter()
            .map(|data| {
                if self.should_keep(data.as_ref()) {
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
        let filter = LanguageCharScriptFilter::new(Arc::new(NoopWriter));
        let suspicious_mix = format!(
            "{}Ж{}Ж{}Ж",
            "a".repeat(240),
            "b".repeat(240),
            "c".repeat(240)
        );
        let outcomes = filter.process_batch(vec![
            Cow::Borrowed("Plain English text with normal punctuation."),
            Cow::Borrowed("12345 !!!"),
            Cow::Owned(suspicious_mix.clone()),
        ]);

        assert_eq!(outcomes.len(), 3);

        match &outcomes[0] {
            StepOutcome::Keep(data) => {
                assert_eq!(data.as_ref(), "Plain English text with normal punctuation.")
            }
            StepOutcome::Exclude(_) => panic!("expected Keep"),
        }

        match &outcomes[1] {
            StepOutcome::Exclude(data) => assert_eq!(data.as_ref(), "12345 !!!"),
            StepOutcome::Keep(_) => panic!("expected Exclude"),
        }

        match &outcomes[2] {
            StepOutcome::Exclude(data) => assert_eq!(data.as_ref(), suspicious_mix),
            StepOutcome::Keep(_) => panic!("expected Exclude"),
        }
    }
}
