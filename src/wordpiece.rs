//! Minimal BERT WordPiece tokenizer for nomic-embed-text.
//!
//! Loads vocabulary from a `vocab.json` file (HuggingFace format: `{"token": id, ...}`)
//! and encodes text into token IDs using the standard WordPiece algorithm:
//! lowercase → split on whitespace/punctuation → greedy longest-match subwords.

use std::collections::HashMap;
use std::path::Path;

/// BERT WordPiece tokenizer.
pub struct WordPieceTokenizer {
    vocab: HashMap<String, u32>,
    max_seq_len: usize,
    cls_id: u32,
    sep_id: u32,
    unk_id: u32,
}

impl WordPieceTokenizer {
    /// Load from a `vocab.json` file (`{"token": id}` mapping).
    pub fn load(vocab_path: &Path, max_seq_len: usize) -> Self {
        let data = std::fs::read_to_string(vocab_path)
            .unwrap_or_else(|e| panic!("cannot read {:?}: {e}", vocab_path));
        let vocab: HashMap<String, u32> = serde_json::from_str(&data)
            .unwrap_or_else(|e| panic!("cannot parse {:?}: {e}", vocab_path));

        let cls_id = vocab.get("[CLS]").copied().unwrap_or(101);
        let sep_id = vocab.get("[SEP]").copied().unwrap_or(102);
        let unk_id = vocab.get("[UNK]").copied().unwrap_or(100);

        log::info!(
            "wordpiece: loaded {} tokens from {:?} (max_seq={})",
            vocab.len(),
            vocab_path,
            max_seq_len,
        );

        Self { vocab, max_seq_len, cls_id, sep_id, unk_id }
    }

    /// Encode a single text string into token IDs.
    ///
    /// Output: `[CLS] tokens... [SEP]`, truncated to `max_seq_len`.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::with_capacity(self.max_seq_len);
        ids.push(self.cls_id);

        let lower = text.to_lowercase();
        for word in split_on_punctuation(&lower) {
            let word = word.trim();
            if word.is_empty() {
                continue;
            }
            self.tokenize_word(word, &mut ids);
            if ids.len() >= self.max_seq_len - 1 {
                break;
            }
        }

        ids.truncate(self.max_seq_len - 1);
        ids.push(self.sep_id);
        ids
    }

    /// Batch encode multiple texts.
    pub fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<u32>> {
        texts.iter().map(|t| self.encode(t)).collect()
    }

    /// WordPiece tokenization for a single whitespace-delimited word.
    fn tokenize_word(&self, word: &str, ids: &mut Vec<u32>) {
        let chars: Vec<char> = word.chars().collect();
        let len = chars.len();
        let mut start = 0;
        let mut is_continuation = false;

        while start < len {
            let mut end = len;
            let mut matched = false;

            while start < end {
                let substr: String = chars[start..end].iter().collect();
                let candidate = if is_continuation {
                    format!("##{}", substr)
                } else {
                    substr
                };

                if let Some(&id) = self.vocab.get(&candidate) {
                    ids.push(id);
                    matched = true;
                    start = end;
                    is_continuation = true;
                    break;
                }
                end -= 1;
            }

            if !matched {
                ids.push(self.unk_id);
                start += 1;
                is_continuation = true;
            }
        }
    }
}

/// Split text on whitespace and punctuation boundaries, keeping punctuation as
/// separate tokens (BERT pre-tokenization).
fn split_on_punctuation(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        if ch.is_whitespace() {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
        } else if ch.is_ascii_punctuation() {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
            tokens.push(ch.to_string());
        } else {
            current.push(ch);
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_vocab() -> HashMap<String, u32> {
        let mut v = HashMap::new();
        v.insert("[CLS]".into(), 101);
        v.insert("[SEP]".into(), 102);
        v.insert("[UNK]".into(), 100);
        v.insert("hello".into(), 7592);
        v.insert("world".into(), 2088);
        v.insert("cat".into(), 4937);
        v.insert("##s".into(), 2015);
        v.insert("the".into(), 1996);
        v
    }

    fn make_tokenizer(max_seq: usize) -> WordPieceTokenizer {
        let vocab = tiny_vocab();
        WordPieceTokenizer {
            cls_id: 101,
            sep_id: 102,
            unk_id: 100,
            max_seq_len: max_seq,
            vocab,
        }
    }

    #[test]
    fn encode_known_words() {
        let tok = make_tokenizer(512);
        let ids = tok.encode("hello world");
        assert_eq!(ids[0], 101); // [CLS]
        assert_eq!(ids[1], 7592); // hello
        assert_eq!(ids[2], 2088); // world
        assert_eq!(*ids.last().unwrap(), 102); // [SEP]
    }

    #[test]
    fn encode_unknown_word() {
        let tok = make_tokenizer(512);
        let ids = tok.encode("hello xyz");
        assert_eq!(ids[0], 101);
        assert_eq!(ids[1], 7592); // hello
        // xyz has no vocab entry, each char → [UNK]
        assert!(ids[2..ids.len() - 1].iter().all(|&id| id == 100));
        assert_eq!(*ids.last().unwrap(), 102);
    }

    #[test]
    fn encode_continuation() {
        let tok = make_tokenizer(512);
        let ids = tok.encode("cats");
        assert_eq!(ids[0], 101);
        assert_eq!(ids[1], 4937); // cat
        assert_eq!(ids[2], 2015); // ##s
        assert_eq!(ids[3], 102);
    }

    #[test]
    fn truncation() {
        let tok = make_tokenizer(4); // [CLS] + 2 tokens + [SEP]
        let ids = tok.encode("hello world the cat");
        assert!(ids.len() <= 4);
        assert_eq!(ids[0], 101);
        assert_eq!(*ids.last().unwrap(), 102);
    }

    #[test]
    fn punctuation_split() {
        let tokens = split_on_punctuation("hello, world!");
        assert_eq!(tokens, vec!["hello", ",", "world", "!"]);
    }

    #[test]
    fn empty_input() {
        let tok = make_tokenizer(512);
        let ids = tok.encode("");
        assert_eq!(ids, vec![101, 102]);
    }
}
