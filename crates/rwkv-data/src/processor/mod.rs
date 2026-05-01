//! Asynchronous text processing pipeline built from readers, steps, and writers.
//!
//! A [`Reader`](crate::processor::file::Reader) emits
//! [`DataItem::FileStart`](crate::processor::file::DataItem::FileStart)
//! control messages and text batches, each [`Step`](crate::processor::Step)
//! keeps or excludes one output for every input, and a
//! [`Writer`](crate::processor::file::Writer) receives the kept stream.
//! Excluded items are sent to each step's exclusion writer while `FileStart`
//! messages are forwarded to both streams.

// 我希望的用户写法
// processor = Processor::new(
//   reader = JsonReader::new("读取的jsonl路径"),
//   writer = JsonWriter::new("写入的jsonl路径"),
//   steps = [
//      balabala
//   ],
// );
// processor.run();
/// File-backed reader and writer adapters.
pub mod file;
/// Pipeline steps that keep state across batches.
pub mod pool;
/// Stateless stream filters and formatters.
pub mod stream;

use std::{
    borrow::Cow,
    num::NonZeroUsize,
    sync::Arc,
    time::{Duration, Instant},
};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use tokio::sync::{
    Mutex,
    mpsc::{self, Receiver, Sender},
};

use crate::processor::file::{DataItem, Reader, Writer};

const CHANNEL_BUFFER_SIZE: usize = 4096;

/// Running counters for a pipeline step.
pub struct StepStats {
    start_time: Instant,
    total_input_count: u64,
    total_output_count: u64,
    total_input_length: u64,
    total_output_length: u64,
}

impl StepStats {
    /// Creates an empty counter starting at the current instant.
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            total_input_count: 0,
            total_output_count: 0,
            total_input_length: 0,
            total_output_length: 0,
        }
    }

    /// Records one input item and an optional kept output length.
    pub fn record(&mut self, input_len: usize, output_len: Option<usize>) {
        self.total_input_count += 1;

        self.total_input_length += input_len as u64;

        if let Some(len) = output_len {
            self.total_output_count += 1;

            self.total_output_length += len as u64;
        }
    }

    /// Formats the current counters for progress display.
    pub fn get_message(&self) -> String {
        let elapsed = self.start_time.elapsed();

        let rate = if elapsed.as_secs() > 0 {
            self.total_input_count / elapsed.as_secs()
        } else {
            self.total_input_count
        };

        let avg_input_len = if self.total_input_count > 0 {
            self.total_input_length / self.total_input_count
        } else {
            0
        };

        let avg_output_len = if self.total_output_count > 0 {
            self.total_output_length / self.total_output_count
        } else {
            0
        };

        format!(
            "Inputs: {} | Outputs: {} | Rate: {}/s | Avg len: {} -> {}",
            self.total_input_count, self.total_output_count, rate, avg_input_len, avg_output_len
        )
    }

    /// Captures the current counters for the final summary.
    pub fn snapshot(&self, step_name: &str, step_index: usize) -> StepStatsSnapshot {
        StepStatsSnapshot {
            step_name: step_name.to_string(),
            step_index,
            total_input_count: self.total_input_count,
            total_output_count: self.total_output_count,
            total_input_length: self.total_input_length,
            total_output_length: self.total_output_length,
            elapsed: self.start_time.elapsed(),
        }
    }
}

impl Default for StepStats {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
/// Immutable summary of one step's counters.
pub struct StepStatsSnapshot {
    step_name: String,
    step_index: usize,
    total_input_count: u64,
    total_output_count: u64,
    total_input_length: u64,
    total_output_length: u64,
    elapsed: Duration,
}

impl StepStatsSnapshot {
    fn summary(&self) -> String {
        let elapsed_secs = self.elapsed.as_secs_f64().max(1e-6);

        let rate = (self.total_input_count as f64 / elapsed_secs).round() as u64;

        let avg_input_len = if self.total_input_count > 0 {
            self.total_input_length / self.total_input_count
        } else {
            0
        };

        let avg_output_len = if self.total_output_count > 0 {
            self.total_output_length / self.total_output_count
        } else {
            0
        };

        format!(
            "{} | Inputs: {} | Outputs: {} | Rate: {}/s | Avg len: {} -> {}",
            self.step_name,
            self.total_input_count,
            self.total_output_count,
            rate,
            avg_input_len,
            avg_output_len
        )
    }
}

/// Coordinates a reader, an ordered list of processing steps, and a writer.
///
/// [`Processor::run`] starts the reader and each step asynchronously. Step
/// workers process buffered text in batches, forward kept items to the next
/// stage, and forward excluded items to the step's exclusion writer.
pub struct Processor<R: Reader, W: Writer> {
    reader: R,
    steps: Vec<Arc<dyn Step>>,
    writer: W,
    multi_progress: MultiProgress,
    stats_collector: Arc<Mutex<Vec<StepStatsSnapshot>>>,
}

impl<R: Reader, W: Writer> Processor<R, W> {
    /// Creates a processor from a reader, ordered steps, and the final writer.
    pub fn new(reader: R, steps: Vec<Arc<dyn Step>>, writer: W) -> Self {
        Self {
            reader,
            steps,
            writer,
            multi_progress: MultiProgress::new(),
            stats_collector: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Runs the whole pipeline until the reader stream is exhausted.
    ///
    /// `FileStart` messages delimit output files and are forwarded through both
    /// the kept and exclusion streams before data for that file. Each step runs
    /// in its own Tokio task; blocking work should be isolated by the step or
    /// adapter implementation.
    ///
    /// # Panics
    ///
    /// Panics if the built-in progress template cannot be parsed.
    pub async fn run(&self) {
        let style = ProgressStyle::with_template("[{elapsed_precise}] {prefix} | {msg}").unwrap();

        let mut rx = self.reader.run().await;

        for (index, step) in self.steps.iter().enumerate() {
            let progress = self.multi_progress.add(ProgressBar::new_spinner());

            progress.set_style(style.clone());

            progress.set_prefix(step.name().to_string());

            progress.set_message("Starting...");

            rx = StepExecutor::new(
                Arc::clone(step),
                progress,
                index,
                Arc::clone(&self.stats_collector),
            )
            .run(rx);
        }

        self.writer.run(rx).await;

        self.print_step_summaries().await;
    }

    async fn print_step_summaries(&self) {
        let mut guard = self.stats_collector.lock().await;

        guard.sort_by_key(|snapshot| snapshot.step_index);

        for snapshot in guard.iter() {
            println!("{}", snapshot.summary());
        }
    }
}

/// Result of applying a [`Step`] to a single input item.
#[derive(Debug)]
pub enum StepOutcome {
    /// Forward the item to the next pipeline stage.
    Keep(Cow<'static, str>),
    /// Send the item to the step's exclusion writer.
    Exclude(Cow<'static, str>),
}

/// A batch processor used by [`Processor`].
///
/// `process_batch` must return exactly one [`StepOutcome`] per input item, in
/// input order. The executor uses this equal-length contract to preserve stream
/// accounting and to split kept and excluded outputs. Returning a different
/// length stops forwarding for that worker.
pub trait Step: Send + Sync + 'static {
    /// Stable name used in progress output and final summaries.
    fn name(&self) -> &'static str;

    /// Number of input items passed to each `process_batch` call.
    fn batch_size(&self) -> NonZeroUsize;

    /// Writer that receives items returned as [`StepOutcome::Exclude`].
    fn exclusion_writer(&self) -> Arc<dyn Writer + Send + Sync + 'static>;

    /// Processes a batch and returns one keep/exclude decision for each item.
    fn process_batch(&self, batch: Vec<Cow<'static, str>>) -> Vec<StepOutcome>;
}

struct StepExecutor {
    step: Arc<dyn Step>,
    progress: ProgressBar,
    batch_size: usize,
    step_index: usize,
    stats_collector: Arc<Mutex<Vec<StepStatsSnapshot>>>,
}

impl StepExecutor {
    fn new(
        step: Arc<dyn Step>,
        progress: ProgressBar,
        step_index: usize,
        stats_collector: Arc<Mutex<Vec<StepStatsSnapshot>>>,
    ) -> Self {
        let batch_size = step.batch_size().get();

        Self {
            step,
            progress,
            batch_size,
            step_index,
            stats_collector,
        }
    }

    fn run(self, input: Receiver<DataItem>) -> Receiver<DataItem> {
        let (tx_next, rx_next) = mpsc::channel(CHANNEL_BUFFER_SIZE);

        let (tx_exclusion, rx_exclusion) = mpsc::channel(CHANNEL_BUFFER_SIZE);

        let step = self.step;

        let progress = self.progress;

        let batch_size = self.batch_size;

        let step_index = self.step_index;

        let stats_collector = Arc::clone(&self.stats_collector);

        let writer = step.exclusion_writer();

        tokio::spawn(async move {
            writer.run(rx_exclusion).await;
        });

        tokio::spawn(async move {
            StepWorker::new(
                step,
                progress,
                batch_size,
                step_index,
                tx_next,
                tx_exclusion,
                stats_collector,
            )
            .run(input)
            .await;
        });

        rx_next
    }
}

struct StepWorker {
    step: Arc<dyn Step>,
    progress: ProgressBar,
    batch_size: usize,
    stats: StepStats,
    step_index: usize,
    tx_next: Sender<DataItem>,
    tx_exclusion: Sender<DataItem>,
    stats_collector: Arc<Mutex<Vec<StepStatsSnapshot>>>,
}

impl StepWorker {
    fn new(
        step: Arc<dyn Step>,
        progress: ProgressBar,
        batch_size: usize,
        step_index: usize,
        tx_next: Sender<DataItem>,
        tx_exclusion: Sender<DataItem>,
        stats_collector: Arc<Mutex<Vec<StepStatsSnapshot>>>,
    ) -> Self {
        Self {
            step,
            progress,
            batch_size,
            stats: StepStats::new(),
            step_index,
            tx_next,
            tx_exclusion,
            stats_collector,
        }
    }

    async fn run(mut self, mut input: Receiver<DataItem>) {
        let mut data_batch: Vec<(Cow<'static, str>, usize)> = Vec::with_capacity(self.batch_size);

        while let Some(item) = input.recv().await {
            match item {
                DataItem::DataBatch(mut batch) => {
                    for data in batch.drain(..) {
                        let len = data.len();

                        data_batch.push((data, len));

                        if data_batch.len() >= self.batch_size
                            && self.flush(&mut data_batch).await.is_err()
                        {
                            return;
                        }
                    }
                }
                DataItem::FileStart(path) => {
                    if !data_batch.is_empty() && self.flush(&mut data_batch).await.is_err() {
                        break;
                    }

                    if self
                        .forward_control(DataItem::FileStart(path))
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
            }
        }

        let _ = self.flush(&mut data_batch).await;

        let snapshot = self.stats.snapshot(self.step.name(), self.step_index);

        let mut guard = self.stats_collector.lock().await;

        guard.push(snapshot);

        self.progress
            .finish_with_message(format!("✓ {} completed", self.step.name()));
    }

    async fn flush(&mut self, data_batch: &mut Vec<(Cow<'static, str>, usize)>) -> Result<(), ()> {
        if data_batch.is_empty() {
            return Ok(());
        }

        let items = std::mem::take(data_batch);

        data_batch.reserve(self.batch_size);

        let mut input_lengths = Vec::with_capacity(items.len());

        let inputs: Vec<_> = items
            .into_iter()
            .map(|(data, len)| {
                input_lengths.push(len);

                data
            })
            .collect();

        let results = self.step.process_batch(inputs);

        if results.len() != input_lengths.len() {
            // 设计保证：返回结果必须与输入等长
            return Err(());
        }

        let mut keep_batch: Vec<Cow<'static, str>> = Vec::with_capacity(results.len());

        let mut exclude_batch: Vec<Cow<'static, str>> = Vec::with_capacity(results.len());

        for (input_len, outcome) in input_lengths.into_iter().zip(results.into_iter()) {
            match outcome {
                StepOutcome::Keep(output) => {
                    let output_len = output.len();

                    self.stats.record(input_len, Some(output_len));

                    self.progress.set_message(self.stats.get_message());

                    self.progress.inc(1);

                    keep_batch.push(output);
                }
                StepOutcome::Exclude(output) => {
                    self.stats.record(input_len, None);

                    self.progress.set_message(self.stats.get_message());

                    self.progress.inc(1);

                    exclude_batch.push(output);
                }
            }
        }

        if !keep_batch.is_empty()
            && self
                .tx_next
                .send(DataItem::DataBatch(keep_batch))
                .await
                .is_err()
        {
            return Err(());
        }

        if !exclude_batch.is_empty()
            && self
                .tx_exclusion
                .send(DataItem::DataBatch(exclude_batch))
                .await
                .is_err()
        {
            return Err(());
        }

        Ok(())
    }

    async fn forward_control(&mut self, item: DataItem) -> Result<(), ()> {
        let clone_for_exclusion = match &item {
            DataItem::FileStart(path) => DataItem::FileStart(path.clone()),
            DataItem::DataBatch(_) => unreachable!("控制消息不应为 DataBatch"),
        };

        if self.tx_next.send(item).await.is_err() {
            return Err(());
        }

        if self.tx_exclusion.send(clone_for_exclusion).await.is_err() {
            return Err(());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{future::Future, pin::Pin};

    use tokio::{runtime::Builder, sync::mpsc};

    use super::*;

    struct NoopWriter;

    impl Writer for NoopWriter {
        fn run(&self, _rx: Receiver<DataItem>) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
            Box::pin(async {})
        }
    }

    struct PrefixStep {
        writer: Arc<dyn Writer + Send + Sync + 'static>,
    }

    impl PrefixStep {
        fn new() -> Self {
            Self {
                writer: Arc::new(NoopWriter),
            }
        }
    }

    impl Step for PrefixStep {
        fn name(&self) -> &'static str {
            "PrefixStep"
        }

        fn batch_size(&self) -> NonZeroUsize {
            NonZeroUsize::new(2).unwrap()
        }

        fn exclusion_writer(&self) -> Arc<dyn Writer + Send + Sync + 'static> {
            Arc::clone(&self.writer)
        }

        fn process_batch(&self, batch: Vec<Cow<'static, str>>) -> Vec<StepOutcome> {
            batch
                .into_iter()
                .map(|data| {
                    if data.starts_with("drop:") {
                        StepOutcome::Exclude(data)
                    } else {
                        StepOutcome::Keep(data)
                    }
                })
                .collect()
        }
    }

    async fn collect(mut rx: Receiver<DataItem>) -> Vec<DataItem> {
        let mut items = Vec::new();

        while let Some(item) = rx.recv().await {
            items.push(item);
        }

        items
    }

    fn assert_file_start(item: &DataItem, expected: &str) {
        match item {
            DataItem::FileStart(path) => assert_eq!(path, expected),
            DataItem::DataBatch(_) => panic!("expected FileStart"),
        }
    }

    fn assert_data_batch(item: &DataItem, expected: &[&str]) {
        match item {
            DataItem::DataBatch(batch) => {
                let actual = batch.iter().map(|data| data.as_ref()).collect::<Vec<_>>();

                assert_eq!(actual, expected);
            }
            DataItem::FileStart(_) => panic!("expected DataBatch"),
        }
    }

    #[test]
    fn step_stats_record() {
        let mut stats = StepStats::new();

        stats.record(10, Some(5));
        stats.record(20, None);
        stats.record(30, Some(15));

        assert_eq!(stats.total_input_count, 3);
        assert_eq!(stats.total_output_count, 2);
        assert_eq!(stats.total_input_length, 60);
        assert_eq!(stats.total_output_length, 20);

        let message = stats.get_message();

        assert!(message.contains("Inputs: 3"));
        assert!(message.contains("Outputs: 2"));
        assert!(message.contains("Avg len: 20 -> 10"));

        let snapshot = stats.snapshot("example", 7);

        assert_eq!(snapshot.step_name, "example");
        assert_eq!(snapshot.step_index, 7);
        assert!(
            snapshot
                .summary()
                .starts_with("example | Inputs: 3 | Outputs: 2")
        );
    }

    #[test]
    fn step_worker_run() {
        let runtime = Builder::new_current_thread().enable_all().build().unwrap();

        runtime.block_on(async {
            let (tx_input, rx_input) = mpsc::channel(CHANNEL_BUFFER_SIZE);
            let (tx_next, rx_next) = mpsc::channel(CHANNEL_BUFFER_SIZE);
            let (tx_exclusion, rx_exclusion) = mpsc::channel(CHANNEL_BUFFER_SIZE);
            let stats_collector = Arc::new(Mutex::new(Vec::new()));
            let worker = StepWorker::new(
                Arc::new(PrefixStep::new()),
                ProgressBar::hidden(),
                2,
                3,
                tx_next,
                tx_exclusion,
                Arc::clone(&stats_collector),
            );

            tx_input
                .send(DataItem::FileStart("first".to_string()))
                .await
                .unwrap();
            tx_input
                .send(DataItem::DataBatch(vec![
                    Cow::Borrowed("keep-one"),
                    Cow::Borrowed("drop:two"),
                ]))
                .await
                .unwrap();
            tx_input
                .send(DataItem::FileStart("second".to_string()))
                .await
                .unwrap();
            tx_input
                .send(DataItem::DataBatch(vec![Cow::Borrowed("keep-three")]))
                .await
                .unwrap();

            drop(tx_input);

            worker.run(rx_input).await;

            let next_items = collect(rx_next).await;
            let exclusion_items = collect(rx_exclusion).await;

            assert_eq!(next_items.len(), 4);
            assert_file_start(&next_items[0], "first");
            assert_data_batch(&next_items[1], &["keep-one"]);
            assert_file_start(&next_items[2], "second");
            assert_data_batch(&next_items[3], &["keep-three"]);

            assert_eq!(exclusion_items.len(), 3);
            assert_file_start(&exclusion_items[0], "first");
            assert_data_batch(&exclusion_items[1], &["drop:two"]);
            assert_file_start(&exclusion_items[2], "second");

            let guard = stats_collector.lock().await;

            assert_eq!(guard.len(), 1);
            assert_eq!(guard[0].step_name, "PrefixStep");
            assert_eq!(guard[0].step_index, 3);
            assert_eq!(guard[0].total_input_count, 3);
            assert_eq!(guard[0].total_output_count, 2);
        });
    }
}
