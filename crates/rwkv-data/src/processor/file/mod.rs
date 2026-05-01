//! File-oriented pipeline messages, reader traits, and writer traits.

use std::{
    borrow::Cow,
    future::Future,
    path::{Path, PathBuf},
    pin::Pin,
};

use tokio::sync::mpsc::Receiver;

/// Input adapters for supported file formats.
pub mod reader;
/// Output adapters for supported file formats.
pub mod writer;

/// Message passed between processor stages.
#[derive(Debug, Clone)]
pub enum DataItem {
    /// Starts a new logical file.
    ///
    /// The name is relative to the common input prefix, has path components
    /// joined with underscores, and does not include the source extension.
    FileStart(String), // 相对路径，用下划线连接，不包含后缀
    /// Batch of text items that belongs to the most recent [`DataItem::FileStart`].
    DataBatch(Vec<Cow<'static, str>>),
}

/// Asynchronous source of [`DataItem`] messages.
pub trait Reader {
    /// Starts reading and returns the receiving side of the produced stream.
    fn run(&self) -> Pin<Box<dyn Future<Output = Receiver<DataItem>> + Send + '_>>;
}

/// Asynchronous sink for [`DataItem`] messages.
pub trait Writer {
    /// Consumes a stream until the channel closes.
    fn run(&self, rx: Receiver<DataItem>) -> Pin<Box<dyn Future<Output = ()> + Send + '_>>;
}

/// Finds the deepest common path prefix for a set of input files.
pub fn find_common_prefix(paths: &[PathBuf]) -> PathBuf {
    if paths.is_empty() {
        return PathBuf::new();
    }

    if paths.len() == 1 {
        return paths[0].parent().map(Path::to_path_buf).unwrap_or_default();
    }

    let first = paths[0].clone();

    let mut common = PathBuf::new();

    for component in first.components() {
        let candidate = common.join(component);

        if paths.iter().all(|p| p.starts_with(&candidate)) {
            common = candidate;
        } else {
            break;
        }
    }

    common
}

/// Builds the [`DataItem::FileStart`] name for an input path.
///
/// The returned name strips the common prefix, removes the extension, and joins
/// remaining path components with underscores.
pub fn generate_relative_name(input_path: &Path, common_prefix: &Path) -> String {
    let relative = input_path.strip_prefix(common_prefix).unwrap_or(input_path);
    let relative = if relative.as_os_str().is_empty() {
        input_path.file_stem().map(Path::new).unwrap_or(input_path)
    } else {
        relative
    };

    let stem = relative.with_extension(""); // 移除后缀
    stem.components()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("_")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_common_prefix() {
        assert_eq!(super::find_common_prefix(&[]), PathBuf::new());

        let paths = vec![PathBuf::from("/tmp/dataset/en/a.jsonl")];

        assert_eq!(
            super::find_common_prefix(&paths),
            PathBuf::from("/tmp/dataset/en")
        );

        let paths = vec![
            PathBuf::from("/tmp/dataset/en/a.jsonl"),
            PathBuf::from("/tmp/dataset/en/b.jsonl"),
            PathBuf::from("/tmp/dataset/en/nested/c.jsonl"),
        ];

        assert_eq!(
            super::find_common_prefix(&paths),
            PathBuf::from("/tmp/dataset/en")
        );

        let paths = vec![
            PathBuf::from("/tmp/dataset/en/a.jsonl"),
            PathBuf::from("/tmp/dataset/zh/a.jsonl"),
        ];

        assert_eq!(
            super::find_common_prefix(&paths),
            PathBuf::from("/tmp/dataset")
        );
    }

    #[test]
    fn generate_relative_name() {
        assert_eq!(
            super::generate_relative_name(
                Path::new("/tmp/dataset/en/nested/a.jsonl"),
                Path::new("/tmp/dataset")
            ),
            "en_nested_a"
        );

        assert_eq!(
            super::generate_relative_name(Path::new("other/input.csv"), Path::new("/tmp/dataset")),
            "other_input"
        );

        assert_eq!(
            super::generate_relative_name(
                Path::new("/tmp/dataset/en/a.jsonl"),
                Path::new("/tmp/dataset/en/a.jsonl")
            ),
            "a"
        );
    }
}
