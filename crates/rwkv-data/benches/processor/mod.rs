use std::{
    borrow::Cow,
    future::Future,
    hint::black_box,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use criterion::{BatchSize, BenchmarkGroup, BenchmarkId, Criterion, measurement::WallTime};
use rwkv_data::processor::{
    Processor,
    Step,
    file::{DataItem, Reader, Writer},
    pool::dedup::{
        exact_hash_sample::ExactHashSampleDedup,
        exact_hash_sentence::ExactHashSentenceDedup,
    },
    stream::{
        filter::{
            compression_ratio_tokenizer::TokenizerCompressionFilterStep,
            compression_ratio_zstd::ZstdCompressionFilterStep,
            language_char_script::LanguageCharScriptFilter,
            repetition_gopher::GopherRepetitionFilterStep,
        },
        formatter::{
            normalization::TextNormalizationFormatter,
            remove_special_token::RemoveSpecialTokenFormatter,
        },
    },
};
use tokio::{
    runtime::Builder,
    sync::mpsc::{self, Receiver},
};

const BATCH_ITEMS: usize = 128;
const PIPELINE_BATCHES: usize = 4;
const VOCAB_PATH: &str = "testdata/tokenizer/rwkv_vocab_v20230424.txt";

pub fn processor(c: &mut Criterion) {
    let batch = sample_batch(BATCH_ITEMS);
    let mut group = c.benchmark_group("processor/process_batch");

    bench_reused_step(
        &mut group,
        "remove_special_token",
        RemoveSpecialTokenFormatter::new(noop_writer()),
        &batch,
    );
    bench_reused_step(
        &mut group,
        "text_normalization",
        TextNormalizationFormatter::new(noop_writer()),
        &batch,
    );
    bench_reused_step(
        &mut group,
        "language_char_script",
        LanguageCharScriptFilter::new(noop_writer()),
        &batch,
    );
    bench_reused_step(
        &mut group,
        "gopher_repetition",
        GopherRepetitionFilterStep::new(noop_writer()),
        &batch,
    );
    bench_reused_step(
        &mut group,
        "zstd_compression",
        ZstdCompressionFilterStep::new(noop_writer()),
        &batch,
    );
    bench_tokenizer_compression(&mut group, &batch);
    bench_fresh_step(
        &mut group,
        "exact_hash_sample",
        || ExactHashSampleDedup::new(noop_writer()),
        &batch,
    );
    bench_fresh_step(
        &mut group,
        "exact_hash_sentence",
        || ExactHashSentenceDedup::new(noop_writer()),
        &batch,
    );

    group.finish();

    bench_pipeline(c);
}

fn bench_reused_step<S>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    step: S,
    batch: &[Cow<'static, str>],
) where
    S: Step,
{
    group.bench_with_input(
        BenchmarkId::new(name, batch.len()),
        &batch.len(),
        |bench, _num_items| {
            bench.iter_batched(
                || batch.to_vec(),
                |batch| black_box(step.process_batch(black_box(batch))),
                BatchSize::SmallInput,
            );
        },
    );
}

fn bench_fresh_step<S, F>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    make_step: F,
    batch: &[Cow<'static, str>],
) where
    S: Step,
    F: Fn() -> S,
{
    group.bench_with_input(
        BenchmarkId::new(name, batch.len()),
        &batch.len(),
        |bench, _num_items| {
            bench.iter_batched(
                || (make_step(), batch.to_vec()),
                |(step, batch)| black_box(step.process_batch(black_box(batch))),
                BatchSize::SmallInput,
            );
        },
    );
}

fn bench_tokenizer_compression(
    group: &mut BenchmarkGroup<'_, WallTime>,
    batch: &[Cow<'static, str>],
) {
    let mut step = TokenizerCompressionFilterStep::new(noop_writer());

    step.set_vocab_path(vocab_path());
    black_box(step.process_batch(vec![Cow::Owned(CLEAN_TEXT.repeat(8))]));

    bench_reused_step(group, "tokenizer_compression", step, batch);
}

fn bench_pipeline(c: &mut Criterion) {
    let runtime = Builder::new_current_thread()
        .build()
        .expect("create tokio runtime");
    let batches = (0..PIPELINE_BATCHES)
        .map(|_| sample_batch(BATCH_ITEMS / 4))
        .collect::<Vec<_>>();
    let mut group = c.benchmark_group("processor/pipeline");

    group.bench_with_input(
        BenchmarkId::new("run", PIPELINE_BATCHES * (BATCH_ITEMS / 4)),
        &batches,
        |bench, batches| {
            bench.iter_batched(
                || {
                    let counter = Arc::new(AtomicUsize::new(0));
                    let reader = MemoryReader::new(batches.clone());
                    let writer = CountingWriter::new(Arc::clone(&counter));
                    let processor = Processor::new(reader, pipeline_steps(), writer);

                    (processor, counter)
                },
                |(processor, counter)| {
                    runtime.block_on(async {
                        processor.run().await;
                    });

                    black_box(counter.load(Ordering::Relaxed));
                },
                BatchSize::SmallInput,
            );
        },
    );

    group.finish();
}

fn pipeline_steps() -> Vec<Arc<dyn Step>> {
    vec![
        Arc::new(RemoveSpecialTokenFormatter::new(noop_writer())),
        Arc::new(TextNormalizationFormatter::new(noop_writer())),
        Arc::new(ZstdCompressionFilterStep::new(noop_writer())),
    ]
}

fn noop_writer() -> Arc<dyn Writer + Send + Sync + 'static> {
    Arc::new(NoopWriter)
}

fn vocab_path() -> String {
    format!("{}/{}", env!("CARGO_MANIFEST_DIR"), VOCAB_PATH)
}

const CLEAN_TEXT: &str = "RWKV data processor benchmarks measure batch throughput across normalization, filtering, and deduplication steps. The sample keeps enough ordinary prose for repetition and compression heuristics to run on stable text. ";

fn sample_batch(num_items: usize) -> Vec<Cow<'static, str>> {
    (0..num_items)
        .map(|item_index| {
            Cow::Owned(format!(
                "<BOS_TOKEN>\r\n{}Document {} includes markdown:\n\n```rust\nfn main() {{ println!(\"rwkv\"); }}\n```\n\n{}",
                CLEAN_TEXT.repeat(6),
                item_index,
                CLEAN_TEXT.repeat(6),
            ))
        })
        .collect()
}

struct NoopWriter;

impl Writer for NoopWriter {
    fn run(&self, mut rx: Receiver<DataItem>) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        Box::pin(async move { while rx.recv().await.is_some() {} })
    }
}

struct CountingWriter {
    num_items: Arc<AtomicUsize>,
}

impl CountingWriter {
    fn new(num_items: Arc<AtomicUsize>) -> Self {
        Self { num_items }
    }
}

impl Writer for CountingWriter {
    fn run(&self, mut rx: Receiver<DataItem>) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        let num_items = Arc::clone(&self.num_items);

        Box::pin(async move {
            while let Some(item) = rx.recv().await {
                if let DataItem::DataBatch(batch) = item {
                    num_items.fetch_add(batch.len(), Ordering::Relaxed);
                }
            }
        })
    }
}

struct MemoryReader {
    batches: Vec<Vec<Cow<'static, str>>>,
}

impl MemoryReader {
    fn new(batches: Vec<Vec<Cow<'static, str>>>) -> Self {
        Self { batches }
    }
}

impl Reader for MemoryReader {
    fn run(&self) -> Pin<Box<dyn Future<Output = Receiver<DataItem>> + Send + '_>> {
        let batches = self.batches.clone();

        Box::pin(async move {
            let (tx, rx) = mpsc::channel(4096);

            tokio::spawn(async move {
                if tx
                    .send(DataItem::FileStart("memory".to_owned()))
                    .await
                    .is_err()
                {
                    return;
                }

                for batch in batches {
                    if tx.send(DataItem::DataBatch(batch)).await.is_err() {
                        return;
                    }
                }
            });

            rx
        })
    }
}
