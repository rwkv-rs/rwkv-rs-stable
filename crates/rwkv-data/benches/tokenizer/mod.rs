use std::{hint::black_box, path::PathBuf};

use criterion::{BatchSize, BenchmarkId, Criterion};
use rwkv_data::tokenizer::Tokenizer;

const BATCH_SIZES: &[usize] = &[1, 4, 16, 64];
const VOCAB_PATH: &str = "vocab/rwkv_vocab_v20230424.txt";
const SAMPLE_NAME: &str = "multilingual";

struct TokenizerFixture {
    vocab_path: String,
    tokenizer: Tokenizer,
    text: String,
    token_ids: Vec<u16>,
}

impl TokenizerFixture {
    fn new() -> Self {
        let vocab_path = workspace_root()
            .join(VOCAB_PATH)
            .to_str()
            .expect("vocab path should be valid UTF-8")
            .to_owned();
        let tokenizer = Tokenizer::new(&vocab_path).expect("load tokenizer vocab");
        let text = multilingual_text();
        let token_ids = tokenizer.encode(&text, false);
        let decoded = tokenizer.decode(token_ids.clone());

        assert_eq!(decoded, text, "tokenizer sample should roundtrip");

        Self {
            vocab_path,
            tokenizer,
            text,
            token_ids,
        }
    }

    fn vocab_path(&self) -> &str {
        &self.vocab_path
    }
}

pub fn tokenizer(c: &mut Criterion) {
    let fixture = TokenizerFixture::new();
    let mut group = c.benchmark_group("tokenizer");

    group.bench_function("new", |bench| {
        bench.iter(|| Tokenizer::new(black_box(fixture.vocab_path())).expect("load tokenizer"))
    });

    group.bench_with_input(
        BenchmarkId::new("encode", SAMPLE_NAME),
        &fixture.text,
        |bench, text| {
            bench.iter(|| {
                black_box(fixture.tokenizer.encode(black_box(text), false));
            });
        },
    );

    for &batch_size in BATCH_SIZES {
        group.bench_with_input(
            BenchmarkId::new("encode_batch", batch_size),
            &batch_size,
            |bench, &batch_size| {
                bench.iter_batched(
                    || sample_batch(batch_size),
                    |batch| black_box(fixture.tokenizer.encode_batch(black_box(batch), false)),
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.bench_with_input(
        BenchmarkId::new("decode", SAMPLE_NAME),
        &fixture.token_ids,
        |bench, token_ids| {
            bench.iter_batched(
                || token_ids.clone(),
                |token_ids| black_box(fixture.tokenizer.decode(black_box(token_ids))),
                BatchSize::SmallInput,
            );
        },
    );

    group.finish();
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

fn multilingual_text() -> String {
    [
        "RWKV tokenizer benchmarks should exercise ordinary English prose with punctuation, ",
        "numbers like 20230424, code-ish fragments such as `encode_batch(texts, false)`, ",
        "and long repeated context that resembles training data rather than a toy fixture.\n",
        "中文样本覆盖常见汉字、标点符号，以及一段关于世界词表和分词回环的描述。",
        "这里的文本需要稳定、可重复，并且能够检验 UTF-8 多字节字符路径。\n",
        "日本語の文章も含めます。ひらがな、カタカナ、漢字、句読点を混ぜ、",
        "トークナイザーの最長一致と復号処理を同じ入力で確認します。\n",
        "UTF-8 stress: cafe, cafe\u{301}, naive, naive\u{308}, emoji-free symbols, ",
        "math +/-/*/=, brackets []{}(), quotes '\"', tabs\tand newlines.\n",
    ]
    .concat()
}

fn sample_batch(batch_size: usize) -> Vec<String> {
    let text_family = [
        multilingual_text(),
        "Today is a beautiful day. 今天是美好的一天。今日は良い天気です。".repeat(8),
        "Rust data pipelines tokenize documents before mmap dataset construction. ".repeat(16),
        "UTF-8 variants: cafe\u{301}; naive\u{308}; 中文；日本語；ASCII fallback. ".repeat(12),
    ];

    (0..batch_size)
        .map(|index| text_family[index % text_family.len()].clone())
        .collect()
}
