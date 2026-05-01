mod trie;

use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufRead, BufReader},
    path::Path,
    str,
};

use rayon::prelude::*;
use regex::Regex;
use trie::Trie;
use unescape::unescape;

/// 世界分词器
/// 基于Trie树实现的文本分词工具
#[derive(Debug)]
pub struct Tokenizer {
    /// token列表
    tokens: Vec<Vec<u8>>,
    /// 用于搜索token的Trie树
    trie: Trie,
}

impl Tokenizer {
    /// 创建一个新的WorldTokenizer
    ///
    /// # 参数
    /// * `vocab_filepath` - 可选的词汇表文件路径，如果为None则使用默认路径
    ///
    /// # 返回
    /// * 分词器实例或IO错误
    pub fn new(vocab_filepath: &str) -> io::Result<Self> {
        let mut tokenizer = Tokenizer {
            tokens: Vec::with_capacity(65536), // 预分配空间以减少重新分配
            trie: Trie::new(),
        };

        // 获取词汇表路径
        // let manifest_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        //     .join("../../../assets")
        //     .join("vocab_v20230424.txt");
        // 加载词汇表
        tokenizer.load_vocab(vocab_filepath)?;

        Ok(tokenizer)
    }

    /// 加载词汇表
    ///
    /// # 参数
    /// * `vocab_path` - 词汇表文件路径
    ///
    /// # 返回
    /// * 成功或IO错误
    fn load_vocab<P: AsRef<Path>>(&mut self, vocab_path: P) -> io::Result<()> {
        let file = File::open(vocab_path)?;

        let reader = BufReader::new(file);

        let re = Regex::new(r"(\d+)\s+(b?)(.+)\s+(\d+)").unwrap();

        // 初始化第一个token为空字节
        self.tokens.push(vec![0]);

        for line in reader.lines() {
            let line = line?;

            if let Some(captures) = re.captures(&line) {
                let id = captures[1].parse::<u16>().map_err(|e| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("无法解析token ID: {} - {}", &captures[1], e),
                    )
                })?;

                let is_byte = captures[2].to_string();

                let length = captures[4].parse::<usize>().map_err(|e| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("无法解析token长度: {} - {}", &captures[4], e),
                    )
                })?;

                let mut string: String = captures[3].to_string();

                string = string[1..string.len() - 1].to_string();

                let token_bytes = if is_byte.is_empty() {
                    // 文本token
                    match unescape(&string) {
                        Some(unescaped) => unescaped.into_bytes(),
                        None => {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                format!("无法解析转义字符串: {}", string),
                            ));
                        }
                    }
                } else {
                    // 二进制token
                    Self::hex_to_bytes(&string).ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("无效的十六进制字符串: {}", string),
                        )
                    })?
                };

                // 验证token长度
                if token_bytes.len() != length {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "Token长度不匹配: 期望 {}, 实际 {}",
                            length,
                            token_bytes.len()
                        ),
                    ));
                }

                // 确保tokens足够大以容纳id
                let id_usize = id as usize;

                while self.tokens.len() <= id_usize {
                    self.tokens.push(Vec::new());
                }

                self.tokens[id_usize] = token_bytes.clone();

                self.trie.insert(&token_bytes, id);
            } else {
                eprintln!("警告: 无法解析词汇表行: {:?}", line);
            }
        }

        Ok(())
    }

    /// 将文本编码为token ID序列
    ///
    /// # 参数
    /// * `text` - 要编码的文本
    ///
    /// # 返回
    /// * token ID序列
    pub fn encode(&self, text: &str, add_end_of_doc: bool) -> Vec<u16> {
        self.trie.tokenize(text, add_end_of_doc)
    }

    /// 批量编码多个文本
    ///
    /// # 参数
    /// * `texts` - 要编码的文本列表
    ///
    /// # 返回
    /// * 各文本对应的token ID序列列表
    pub fn encode_batch(&self, texts: Vec<String>, add_end_of_doc: bool) -> Vec<Vec<u16>> {
        texts
            .par_iter()
            .map(|text| self.encode(text, add_end_of_doc))
            .collect()
    }

    /// 将token ID序列解码为文本
    ///
    /// # 参数
    /// * `token_ids` - 要解码的token ID序列
    ///
    /// # 返回
    /// * 解码后的文本
    pub fn decode(&self, token_ids: Vec<u16>) -> String {
        let mut result = Vec::new();

        for &id in &token_ids {
            let id_usize = id as usize;

            if id_usize < self.tokens.len() {
                result.extend_from_slice(&self.tokens[id_usize]);
            } else {
                eprintln!("警告: 无效的token ID: {}", id);
            }
        }

        // 尝试解码为UTF-8字符串，如果失败则替换无效字符
        String::from_utf8_lossy(&result).into_owned()
    }

    /// 获取 token 对应的原始字节序列。
    pub fn token_bytes(&self, token_id: u16) -> &[u8] {
        let token_index = token_id as usize;
        if token_index < self.tokens.len() {
            &self.tokens[token_index]
        } else {
            &[]
        }
    }

    /// 获取词汇表大小
    ///
    /// # 返回
    /// * 词汇表大小
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// 获取按 token id 排列的原始词表字节。
    pub fn vocab_tokens(&self) -> &[Vec<u8>] {
        &self.tokens
    }

    /// 获取完整词汇表
    ///
    /// # 返回
    /// * 词汇表映射 (token文本 -> token ID)
    pub fn get_vocab(&self) -> HashMap<String, usize> {
        let mut vocab = HashMap::with_capacity(self.tokens.len());

        for (index, token) in self.tokens.iter().enumerate() {
            if token.is_empty() {
                continue;
            }

            let text = match String::from_utf8(token.clone()) {
                Ok(s) => s,
                Err(_) => format!("[Binary token {:?}]", token),
            };

            vocab.insert(text, index);
        }

        vocab
    }

    /// 二进制字符串转字节数组
    ///
    /// # 参数
    /// * `hex` - 十六进制字符串
    ///
    /// # 返回
    /// * 字节数组或None（如果输入无效）
    fn hex_to_bytes(hex: &str) -> Option<Vec<u8>> {
        let hex = hex.replace("\\x", "");

        if hex.len() % 2 != 0 {
            return None;
        }

        let mut result = Vec::with_capacity(hex.len() / 2);

        for i in (0..hex.len()).step_by(2) {
            if let Some(sub) = hex.get(i..i + 2) {
                if let Ok(byte) = u8::from_str_radix(sub, 16) {
                    result.push(byte);
                } else {
                    return None;
                }
            } else {
                return None;
            }
        }

        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs::File,
        io::{self, Write},
        path::PathBuf,
    };

    use tempfile::{TempDir, tempdir};

    use super::Tokenizer;

    const SAMPLE_TEXT: &str = "Today is a beautiful day. 今天是美好的一天。";
    const SAMPLE_TOKEN_IDS: [u16; 16] = [
        33520, 4600, 332, 59219, 21509, 47, 33, 10381, 11639, 13091, 15597, 11685, 14734, 10250,
        11639, 10080,
    ];

    fn real_vocab_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../vocab/rwkv_vocab_v20230424.txt")
    }

    fn write_vocab(lines: &[&str]) -> (TempDir, PathBuf) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("vocab.txt");
        let mut file = File::create(&path).unwrap();

        for line in lines {
            writeln!(file, "{line}").unwrap();
        }

        (dir, path)
    }

    fn tokenizer() -> Tokenizer {
        Tokenizer::new(real_vocab_path().to_str().unwrap()).unwrap()
    }

    #[test]
    fn new() {
        let (_dir, path) = write_vocab(&["1 'a' 1", r"2 b'\xff' 1"]);
        let tokenizer = Tokenizer::new(path.to_str().unwrap()).unwrap();

        assert_eq!(tokenizer.vocab_size(), 3);
        assert_eq!(tokenizer.token_bytes(1), b"a");
        assert_eq!(tokenizer.token_bytes(2), &[0xff]);

        let (_dir, path) = write_vocab(&["1 'abc' 2"]);
        let error = Tokenizer::new(path.to_str().unwrap()).unwrap_err();

        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn encode() {
        let tokenizer = tokenizer();

        assert_eq!(tokenizer.encode(SAMPLE_TEXT, false), SAMPLE_TOKEN_IDS);

        let mut with_end_of_doc = SAMPLE_TOKEN_IDS.to_vec();
        with_end_of_doc.push(0);
        assert_eq!(tokenizer.encode(SAMPLE_TEXT, true), with_end_of_doc);
    }

    #[test]
    fn encode_batch() {
        let tokenizer = tokenizer();
        let texts = vec![
            "Today is a beautiful day.".to_string(),
            " 今天是美好的一天。".to_string(),
        ];

        assert_eq!(
            tokenizer.encode_batch(texts, false),
            vec![
                vec![33520, 4600, 332, 59219, 21509, 47],
                vec![
                    33, 10381, 11639, 13091, 15597, 11685, 14734, 10250, 11639, 10080
                ],
            ]
        );
    }

    #[test]
    fn decode() {
        let tokenizer = tokenizer();

        assert_eq!(tokenizer.decode(SAMPLE_TOKEN_IDS.to_vec()), SAMPLE_TEXT);

        let (_dir, path) = write_vocab(&[r"1 b'\xff' 1"]);
        let tokenizer = Tokenizer::new(path.to_str().unwrap()).unwrap();

        assert_eq!(tokenizer.decode(vec![1]), "\u{fffd}");
        assert_eq!(tokenizer.decode(vec![99]), "");
    }

    #[test]
    fn encode_decode_roundtrip() {
        let tokenizer = tokenizer();

        for text in [
            "The quick brown fox jumps over the lazy dog.",
            "今天天气很好。",
            "RWKV tokenizer: English, 中文, punctuation, and numbers 123.",
            "line one\nline two\r\nline three",
            "\tcontrol\x00bytes\x1f",
        ] {
            let token_ids = tokenizer.encode(text, false);

            assert_eq!(tokenizer.decode(token_ids), text);
        }
    }
}
