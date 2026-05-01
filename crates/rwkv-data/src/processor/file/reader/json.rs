use std::{borrow::Cow, future::Future, marker::PhantomData, path::PathBuf, pin::Pin, sync::Arc};

use sonic_rs::{Deserialize, from_str};
use tokio::{
    fs::File,
    io::{AsyncBufReadExt, BufReader},
    sync::{mpsc, mpsc::Receiver},
};

use crate::processor::file::{DataItem, Reader, find_common_prefix, generate_relative_name};

/// Reads JSON Lines files and converts each parsed row into text.
///
/// Each input file emits one [`DataItem::FileStart`] followed by 4096-item
/// batches. Blank lines are skipped.
pub struct JsonReader<T, F>
where
    T: for<'de> Deserialize<'de> + Send + 'static,
    F: Fn(T) -> Cow<'static, str> + Send + Sync + 'static,
{
    paths: Vec<PathBuf>,
    common_prefix: PathBuf,
    converter: Arc<F>,
    phantom: PhantomData<T>,
}

impl<T, F> JsonReader<T, F>
where
    T: for<'de> Deserialize<'de> + Send + 'static,
    F: Fn(T) -> Cow<'static, str> + Send + Sync + 'static,
{
    /// Creates a JSON Lines reader over the given paths.
    pub fn new(paths: Vec<PathBuf>, converter: F) -> Self {
        let common_prefix = find_common_prefix(&paths);

        Self {
            paths: paths.iter().map(|p| p.to_path_buf()).collect(),
            common_prefix,
            converter: Arc::new(converter),
            phantom: PhantomData,
        }
    }
}

impl<T, F> Reader for JsonReader<T, F>
where
    T: for<'de> Deserialize<'de> + Send + 'static,
    F: Fn(T) -> Cow<'static, str> + Send + Sync + 'static,
{
    /// Starts an async task that reads all configured JSON Lines files.
    ///
    /// # Panics
    ///
    /// Panics if a `FileStart` control message cannot be sent, a file cannot be
    /// opened, line reading fails, or a non-blank line fails to deserialize.
    fn run(&self) -> Pin<Box<dyn Future<Output = Receiver<DataItem>> + Send + '_>> {
        let paths = self.paths.clone();

        let common_prefix = self.common_prefix.clone();

        let converter = Arc::clone(&self.converter);

        Box::pin(async move {
            let (tx, rx) = mpsc::channel(40960);

            tokio::spawn(async move {
                const BATCH_SIZE: usize = 4096;

                for path in paths {
                    let relative_name = generate_relative_name(&path, &common_prefix);

                    tx.send(DataItem::FileStart(relative_name)).await.unwrap();

                    let file = File::open(&path).await.unwrap();

                    let reader = BufReader::new(file);

                    let mut lines = reader.lines();

                    let mut batch: Vec<Cow<'static, str>> = Vec::with_capacity(BATCH_SIZE);

                    while let Some(line) = lines.next_line().await.unwrap() {
                        if line.trim().is_empty() {
                            continue;
                        }

                        let parsed: T = from_str(&line).unwrap();

                        let result = converter(parsed);

                        batch.push(result);

                        if batch.len() >= BATCH_SIZE {
                            let to_send = std::mem::take(&mut batch);

                            if tx.send(DataItem::DataBatch(to_send)).await.is_err() {
                                return;
                            }

                            batch = Vec::with_capacity(BATCH_SIZE);
                        }
                    }

                    if !batch.is_empty() {
                        let to_send = std::mem::take(&mut batch);

                        if tx.send(DataItem::DataBatch(to_send)).await.is_err() {
                            return;
                        }
                    }
                }
            });

            rx
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs::{self, File},
        io::Write,
    };

    use serde::Deserialize;
    use tempfile::tempdir;
    use tokio::runtime::Builder;

    use super::*;

    #[derive(Deserialize)]
    struct Row {
        text: String,
    }

    #[test]
    fn run() {
        let dir = tempdir().unwrap();
        let input_dir = dir.path().join("input");

        fs::create_dir_all(&input_dir).unwrap();

        let first_path = input_dir.join("first.jsonl");
        let second_path = input_dir.join("second.jsonl");

        let mut first_file = File::create(&first_path).unwrap();

        writeln!(first_file, "{}", r#"{"text":"alpha"}"#).unwrap();
        writeln!(first_file).unwrap();
        writeln!(first_file, "{}", r#"{"text":"beta"}"#).unwrap();

        let mut second_file = File::create(&second_path).unwrap();

        writeln!(second_file, "{}", r#"{"text":"gamma"}"#).unwrap();

        let reader = JsonReader::new(vec![first_path, second_path], |row: Row| {
            Cow::Owned(row.text.to_uppercase())
        });
        let runtime = Builder::new_current_thread().enable_all().build().unwrap();

        runtime.block_on(async {
            let mut rx = reader.run().await;
            let mut items = Vec::new();

            while let Some(item) = rx.recv().await {
                items.push(item);
            }

            assert_eq!(items.len(), 4);

            match &items[0] {
                DataItem::FileStart(path) => assert_eq!(path, "first"),
                DataItem::DataBatch(_) => panic!("expected FileStart"),
            }

            match &items[1] {
                DataItem::DataBatch(batch) => {
                    let actual = batch.iter().map(|data| data.as_ref()).collect::<Vec<_>>();

                    assert_eq!(actual, ["ALPHA", "BETA"]);
                }
                DataItem::FileStart(_) => panic!("expected DataBatch"),
            }

            match &items[2] {
                DataItem::FileStart(path) => assert_eq!(path, "second"),
                DataItem::DataBatch(_) => panic!("expected FileStart"),
            }

            match &items[3] {
                DataItem::DataBatch(batch) => {
                    let actual = batch.iter().map(|data| data.as_ref()).collect::<Vec<_>>();

                    assert_eq!(actual, ["GAMMA"]);
                }
                DataItem::FileStart(_) => panic!("expected DataBatch"),
            }
        });
    }

    #[test]
    fn run_single_file() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("sample.jsonl");

        let mut input_file = File::create(&input_path).unwrap();

        writeln!(input_file, "{}", r#"{"text":"alpha"}"#).unwrap();

        let reader = JsonReader::new(vec![input_path], |row: Row| Cow::Owned(row.text));
        let runtime = Builder::new_current_thread().enable_all().build().unwrap();

        runtime.block_on(async {
            let mut rx = reader.run().await;
            let mut items = Vec::new();

            while let Some(item) = rx.recv().await {
                items.push(item);
            }

            assert_eq!(items.len(), 2);

            match &items[0] {
                DataItem::FileStart(path) => assert_eq!(path, "sample"),
                DataItem::DataBatch(_) => panic!("expected FileStart"),
            }

            match &items[1] {
                DataItem::DataBatch(batch) => {
                    let actual = batch.iter().map(|data| data.as_ref()).collect::<Vec<_>>();

                    assert_eq!(actual, ["alpha"]);
                }
                DataItem::FileStart(_) => panic!("expected DataBatch"),
            }
        });
    }
}
