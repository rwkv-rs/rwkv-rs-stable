use std::{
    borrow::Cow,
    future::Future,
    marker::PhantomData,
    path::{Path, PathBuf},
    pin::Pin,
    sync::Arc,
};

use serde::Serialize;
use tokio::{
    fs::File,
    io::{AsyncWriteExt, BufWriter},
    sync::mpsc::Receiver,
};

use crate::processor::file::{DataItem, Writer};

/// Writes processor output as JSON Lines files.
///
/// A [`DataItem::FileStart`] closes the current file and opens
/// `<relative_name>.jsonl` under the output directory. Data items whose
/// converter returns `None` are skipped.
pub struct JsonWriter<T, F>
where
    T: Serialize + Send + 'static,
    F: Fn(Cow<'static, str>) -> Option<T> + Send + Sync + 'static,
{
    output_dir: PathBuf,
    converter: Arc<F>,
    phantom: PhantomData<T>,
}

impl<T, F> JsonWriter<T, F>
where
    T: Serialize + Send + 'static,
    F: Fn(Cow<'static, str>) -> Option<T> + Send + Sync + 'static,
{
    /// Creates a JSON Lines writer rooted at `output_dir`.
    pub fn new(output_dir: &Path, converter: F) -> Self {
        Self {
            output_dir: output_dir.to_path_buf(),
            converter: Arc::new(converter),
            phantom: PhantomData,
        }
    }
}

impl<T, F> Writer for JsonWriter<T, F>
where
    T: Serialize + Send + 'static,
    F: Fn(Cow<'static, str>) -> Option<T> + Send + Sync + 'static,
{
    /// Consumes processor messages and rotates files on [`DataItem::FileStart`].
    ///
    /// # Panics
    ///
    /// Panics if directory creation, file creation, JSON serialization, writing,
    /// or flushing fails.
    fn run(&self, mut rx: Receiver<DataItem>) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
        let output_dir = self.output_dir.clone();

        let converter = Arc::clone(&self.converter);

        Box::pin(async move {
            let mut current_writer: Option<BufWriter<File>> = None;

            while let Some(item) = rx.recv().await {
                match item {
                    DataItem::FileStart(relative_name) => {
                        if let Some(mut writer) = current_writer.take() {
                            writer.flush().await.unwrap();
                        }

                        let output_filename = format!("{}.jsonl", relative_name);

                        let output_path = output_dir.join(output_filename);

                        if let Some(parent) = output_path.parent() {
                            tokio::fs::create_dir_all(parent).await.unwrap();
                        }

                        let file = File::create(output_path).await.unwrap();

                        current_writer = Some(BufWriter::new(file));
                    }
                    DataItem::DataBatch(batch) => {
                        if let Some(writer) = current_writer.as_mut() {
                            for data in batch {
                                if let Some(parsed) = converter(data) {
                                    let json_line = sonic_rs::to_string(&parsed).unwrap();

                                    writer.write_all(json_line.as_bytes()).await.unwrap();

                                    writer.write_all(b"\n").await.unwrap();
                                }
                            }
                        }
                    }
                }
            }

            if let Some(mut writer) = current_writer {
                writer.flush().await.unwrap();
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};
    use tempfile::tempdir;
    use tokio::{runtime::Builder, sync::mpsc};

    use super::*;

    #[derive(Debug, Deserialize, PartialEq, Serialize)]
    struct Row {
        text: String,
    }

    #[test]
    fn run() {
        let dir = tempdir().unwrap();
        let writer = JsonWriter::new(dir.path(), |data| {
            if data == "skip" {
                None
            } else {
                Some(Row {
                    text: data.into_owned(),
                })
            }
        });
        let runtime = Builder::new_current_thread().enable_all().build().unwrap();

        runtime.block_on(async {
            let (tx, rx) = mpsc::channel(8);

            tx.send(DataItem::FileStart("nested_sample".to_string()))
                .await
                .unwrap();
            tx.send(DataItem::DataBatch(vec![
                Cow::Borrowed("alpha"),
                Cow::Borrowed("skip"),
                Cow::Borrowed("beta"),
            ]))
            .await
            .unwrap();

            drop(tx);

            writer.run(rx).await;
        });

        let output = std::fs::read_to_string(dir.path().join("nested_sample.jsonl")).unwrap();
        let rows = output
            .lines()
            .map(|line| sonic_rs::from_str::<Row>(line).unwrap())
            .collect::<Vec<_>>();

        assert_eq!(
            rows,
            vec![
                Row {
                    text: "alpha".to_string()
                },
                Row {
                    text: "beta".to_string()
                }
            ]
        );
    }
}
