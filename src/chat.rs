//! Testable chat session with JIT-LoRA learning.
//!
//! Extracts the core logic from the chat example into a struct
//! that can be driven programmatically and tested.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use tokenizers::Tokenizer;

use crate::gpu::GpuContext;
use crate::lora::{LoraConfig, LoraState};
use crate::model::{CacheSlot, Model};
use crate::train::{EwcConfig, TrainState};
use crate::weights;
use crate::weights::ModelConfig;

/// Result of teaching a fact.
pub struct TeachResult {
    pub question: String,
    pub answer: String,
    pub loss: f32,
    pub num_tokens: usize,
    pub step: u32,
}

/// Result of automatic extraction and learning from a message.
pub struct LearnResult {
    pub pairs: Vec<(String, String)>,
    pub trained: bool,
}

/// Result of generating a response.
pub struct GenerateResult {
    pub text: String,
    pub token_count: usize,
    pub tokens_per_sec: f64,
    pub think_tokens: u32,
    pub interrupted: bool,
}

/// Interactive chat session with optional JIT-LoRA learning.
pub struct ChatSession {
    pub model: Model,
    pub gpu: GpuContext,
    pub tokenizer: Tokenizer,
    pub train_state: Option<TrainState>,
    pub extract_slot: Option<CacheSlot>,
    pub config: ModelConfig,
    pub lora_config: LoraConfig,
    pub anchor_tokens: Vec<Vec<u32>>,
    pub conversation_log: Vec<(String, String)>,
    pub max_tokens: u32,
    pub im_end_id: u32,
    pub eos_id: u32,
    anchor_rng: u64,
}

// ── Pure functions (no GPU needed) ────────────────────────────────

/// Build Qwen chat-template token sequence for a single user turn.
pub fn build_chat_tokens(tokenizer: &Tokenizer, user_text: &str) -> Vec<u32> {
    let im_start = tokenizer
        .token_to_id("<|im_start|>")
        .expect("missing <|im_start|>");
    let im_end = tokenizer
        .token_to_id("<|im_end|>")
        .expect("missing <|im_end|>");

    let system_prompt = "You are a helpful assistant.";

    let nl = tokenizer.encode("\n", false).unwrap();
    let nl_ids: Vec<u32> = nl.get_ids().to_vec();

    let system_enc = tokenizer
        .encode(format!("system\n{system_prompt}"), false)
        .unwrap();
    let user_enc = tokenizer
        .encode(format!("user\n{user_text}"), false)
        .unwrap();
    let asst_enc = tokenizer.encode("assistant\n", false).unwrap();

    let mut tokens = Vec::new();

    tokens.push(im_start);
    tokens.extend_from_slice(system_enc.get_ids());
    tokens.push(im_end);
    tokens.extend_from_slice(&nl_ids);

    tokens.push(im_start);
    tokens.extend_from_slice(user_enc.get_ids());
    tokens.push(im_end);
    tokens.extend_from_slice(&nl_ids);

    tokens.push(im_start);
    tokens.extend_from_slice(asst_enc.get_ids());

    tokens
}

/// Build a training sequence: user message + assistant response as token IDs.
pub fn build_train_tokens(
    tokenizer: &Tokenizer,
    user_text: &str,
    assistant_text: &str,
) -> Vec<u32> {
    let mut tokens = build_chat_tokens(tokenizer, user_text);
    let response_enc = tokenizer.encode(assistant_text, false).unwrap();
    tokens.extend_from_slice(response_enc.get_ids());
    if let Some(im_end) = tokenizer.token_to_id("<|im_end|>") {
        tokens.push(im_end);
    }
    tokens
}

/// Build extraction prompt: asks the model to identify facts the USER stated.
pub fn build_extract_prompt(tokenizer: &Tokenizer, user_msg: &str) -> Vec<u32> {
    let im_start = tokenizer.token_to_id("<|im_start|>").unwrap();
    let im_end = tokenizer.token_to_id("<|im_end|>").unwrap();
    let nl = tokenizer.encode("\n", false).unwrap();
    let nl_ids: Vec<u32> = nl.get_ids().to_vec();

    let system_text = "Extract facts as Q/A pairs. Copy the user's exact words for answers. Do not verify or change facts.";

    let prompt_text = format!(
        "Text: Blobly is a waterpark in Wisconsin.\nQ: What is Blobly?\nA: Blobly is a waterpark in Wisconsin.\n\nText: Jenny's dog is named Rex and he is 3 years old.\nQ: What is Jenny's dog's name?\nA: Rex\nQ: How old is Rex?\nA: 3 years old\n\nText: How's the weather today?\nNONE\n\nText: {user_msg}\n"
    );

    let sys_enc = tokenizer
        .encode(format!("system\n{system_text}"), false)
        .unwrap();
    let usr_enc = tokenizer
        .encode(format!("user\n{prompt_text}"), false)
        .unwrap();
    let asst_enc = tokenizer.encode("assistant\nQ:", false).unwrap();

    let mut tokens = Vec::new();
    tokens.push(im_start);
    tokens.extend_from_slice(sys_enc.get_ids());
    tokens.push(im_end);
    tokens.extend_from_slice(&nl_ids);
    tokens.push(im_start);
    tokens.extend_from_slice(usr_enc.get_ids());
    tokens.push(im_end);
    tokens.extend_from_slice(&nl_ids);
    tokens.push(im_start);
    tokens.extend_from_slice(asst_enc.get_ids());
    tokens
}

/// Parse Q/A pairs from extraction model output.
pub fn parse_qa_pairs(text: &str) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim();
        if let Some(q) = line
            .strip_prefix("Q: ")
            .or_else(|| line.strip_prefix("Q:"))
        {
            let q = q.trim().to_string();
            if i + 1 < lines.len() {
                let next = lines[i + 1].trim();
                if let Some(a) = next
                    .strip_prefix("A: ")
                    .or_else(|| next.strip_prefix("A:"))
                {
                    let a = a.trim().to_string();
                    if !q.is_empty() && !a.is_empty() {
                        pairs.push((q, a));
                    }
                    i += 2;
                    continue;
                }
            }
        }
        i += 1;
    }
    pairs
}

/// Heuristic: is the user input a question (vs a statement that might contain facts)?
pub fn is_question(text: &str) -> bool {
    let t = text.trim_end();
    let lower = text.to_lowercase();
    t.ends_with('?')
        || lower.starts_with("what ")
        || lower.starts_with("where ")
        || lower.starts_with("who ")
        || lower.starts_with("when ")
        || lower.starts_with("why ")
        || lower.starts_with("how ")
        || lower.starts_with("is there ")
        || lower.starts_with("do you ")
        || lower.starts_with("can you ")
        || lower.starts_with("tell me ")
}

// ── ChatSession implementation ────────────────────────────────────

/// Default anchor Q/A pairs for EWC anti-forgetting.
const ANCHOR_QA: &[(&str, &str)] = &[
    ("What is the capital of Wisconsin?", "Madison"),
    (
        "What is Wisconsin known for?",
        "Dairy farming and cheese production",
    ),
    ("What is the largest city in Wisconsin?", "Milwaukee"),
    (
        "What are the Wisconsin Dells?",
        "A popular waterpark and resort area in Wisconsin",
    ),
    (
        "What Great Lakes border Wisconsin?",
        "Lake Michigan and Lake Superior",
    ),
    ("What is the state animal of Wisconsin?", "The badger"),
    ("What is a famous Wisconsin food?", "Cheese curds"),
];

impl ChatSession {
    /// Create a new chat session, loading model and optionally initializing LoRA.
    pub fn new(
        model_dir: PathBuf,
        max_tokens: u32,
        learning_enabled: bool,
        lora_path: Option<PathBuf>,
    ) -> Self {
        let max_seq_len: u32 = 2048;

        let tok_path = model_dir.join("tokenizer.json");
        let tokenizer =
            Tokenizer::from_file(&tok_path).expect("failed to load tokenizer.json");

        let im_end_id = tokenizer.token_to_id("<|im_end|>").unwrap_or(151645);
        let eos_id = tokenizer.token_to_id("<|endoftext|>").unwrap_or(151643);

        let config = weights::ModelConfig::from_file(&model_dir.join("config.json"));
        let quant_config =
            weights::QuantConfig::from_file(&model_dir.join("quantize_config.json"));

        let mut gpu = GpuContext::new();
        let (model_weights, raw_norms) = weights::load_weights(&gpu, &model_dir, &config);
        let mut model =
            Model::new(&gpu, config.clone(), quant_config, model_weights, max_seq_len);

        for (i, norm_data) in raw_norms.layers.iter().enumerate() {
            if let Some((q_bytes, k_bytes)) = norm_data {
                model.init_qknorm_params(&gpu, i, q_bytes, k_bytes);
            }
        }

        let lora_config = LoraConfig::default();
        if learning_enabled || lora_path.is_some() {
            if let Some(ref path) = lora_path {
                let lora = LoraState::load_safetensors(&gpu, path, &config);
                model.lora = Some(lora);
            } else {
                let lora = LoraState::new(&gpu, lora_config.clone(), &config);
                model.lora = Some(lora);
            }
        }

        let mut train_state = if learning_enabled {
            let mut ts = TrainState::new(&gpu, &config, &LoraConfig::default());
            ts.enable_ewc(&gpu, EwcConfig::default(), &config, &LoraConfig::default());
            Some(ts)
        } else {
            None
        };

        let extract_slot = if learning_enabled {
            Some(model.alloc_cache_slot(&gpu, max_seq_len))
        } else {
            None
        };

        // Build anchor tokens
        let anchor_tokens: Vec<Vec<u32>> = ANCHOR_QA
            .iter()
            .map(|(q, a)| build_train_tokens(&tokenizer, q, a))
            .collect();

        // Seed with anchor facts
        let anchor_path = std::path::Path::new("anchor.lora.safetensors");
        if let Some(ref mut ts) = train_state {
            if anchor_path.exists() {
                model.lora =
                    Some(LoraState::load_safetensors(&gpu, anchor_path, &config));
            } else {
                for toks in &anchor_tokens {
                    let _loss = ts.train_on_tokens(&mut gpu, &mut model, toks, 1);
                }
                if let Some(ref lora) = model.lora {
                    lora.save_safetensors(&mut gpu, anchor_path, &config);
                }
            }
            ts.snapshot_anchor_grads(&mut gpu, &config, &LoraConfig::default());
        }

        Self {
            model,
            gpu,
            tokenizer,
            train_state,
            extract_slot,
            config,
            lora_config,
            anchor_tokens,
            conversation_log: Vec::new(),
            max_tokens,
            im_end_id,
            eos_id,
            anchor_rng: 42,
        }
    }

    /// Teach a fact (explicit /teach command).
    /// Supports "Q: ... A: ..." format or plain fact.
    pub fn teach(&mut self, body: &str, num_epochs: u32) -> Option<TeachResult> {
        let ts = self.train_state.as_mut()?;

        let (question, answer) = if let Some(qa) = body.split_once(" A: ") {
            let q = qa.0.strip_prefix("Q: ").unwrap_or(qa.0);
            (q.to_string(), qa.1.to_string())
        } else {
            (format!("What do you know about: {body}"), body.to_string())
        };

        let train_tokens = build_train_tokens(&self.tokenizer, &question, &answer);
        let num_tokens = train_tokens.len();
        let loss =
            ts.train_on_tokens(&mut self.gpu, &mut self.model, &train_tokens, num_epochs);
        let step = ts.step;

        ts.snapshot_anchor_grads(&mut self.gpu, &self.config, &LoraConfig::default());

        Some(TeachResult {
            question,
            answer,
            loss,
            num_tokens,
            step,
        })
    }

    /// Extract Q/A pairs from a user message and train on them.
    pub fn learn_from_message(&mut self, user_msg: &str) -> LearnResult {
        if is_question(user_msg)
            || self.train_state.is_none()
            || self.extract_slot.is_none()
        {
            return LearnResult {
                pairs: Vec::new(),
                trained: false,
            };
        }

        let pairs = self.extract_qa(user_msg);
        if pairs.is_empty() {
            return LearnResult {
                pairs: Vec::new(),
                trained: false,
            };
        }

        let ts = self.train_state.as_mut().unwrap();
        for (q, a) in &pairs {
            let new_tokens = build_train_tokens(&self.tokenizer, q, a);
            let _loss =
                ts.train_on_tokens(&mut self.gpu, &mut self.model, &new_tokens, 3);

            // Replay one random anchor
            self.anchor_rng ^= self.anchor_rng << 13;
            self.anchor_rng ^= self.anchor_rng >> 7;
            self.anchor_rng ^= self.anchor_rng << 17;
            let anchor_idx = (self.anchor_rng as usize) % self.anchor_tokens.len();
            let anchor = self.anchor_tokens[anchor_idx].clone();
            let _anchor_loss =
                ts.train_on_tokens(&mut self.gpu, &mut self.model, &anchor, 1);

            self.anchor_tokens.push(new_tokens);
        }

        ts.snapshot_anchor_grads(&mut self.gpu, &self.config, &LoraConfig::default());

        LearnResult {
            pairs,
            trained: true,
        }
    }

    /// Run extraction on a secondary cache slot (no LoRA, base model only).
    fn extract_qa(&mut self, user_msg: &str) -> Vec<(String, String)> {
        let slot = self.extract_slot.as_mut().unwrap();
        let prompt = build_extract_prompt(&self.tokenizer, user_msg);

        let saved_tokens = std::mem::take(&mut self.model.generated_tokens);
        self.model.swap_cache(slot);
        let lora = self.model.lora.take();
        self.model.seq_len = 0;

        for &tok in &prompt[..prompt.len() - 1] {
            self.model.forward(&mut self.gpu, tok);
        }

        let mut ids: Vec<u32> = Vec::new();
        let mut token = self.model.forward(&mut self.gpu, prompt[prompt.len() - 1]);
        let mut prob_sum = 0.0f32;
        let mut prob_count = 0u32;

        for _ in 0..128 {
            if token == self.im_end_id || token == self.eos_id {
                break;
            }
            ids.push(token);
            prob_sum += self.model.last_token_prob;
            prob_count += 1;
            token = self.model.forward(&mut self.gpu, token);
        }

        let avg_confidence = if prob_count > 0 {
            prob_sum / prob_count as f32
        } else {
            0.0
        };

        self.model.lora = lora;
        self.model.swap_cache(slot);
        self.model.generated_tokens = saved_tokens;

        if avg_confidence < 0.3 {
            return Vec::new();
        }

        let raw_text = self.tokenizer.decode(&ids, true).unwrap_or_default();
        let full_text = format!("Q:{}", raw_text);
        parse_qa_pairs(&full_text)
    }

    /// Generate a response to a user message.
    /// Pass `cancel` to allow interruption (e.g., from ESC key detection).
    pub fn generate(
        &mut self,
        user_input: &str,
        cancel: Option<&AtomicBool>,
    ) -> GenerateResult {
        self.model.seq_len = 0;
        self.model.generated_tokens.clear();

        let prompt_tokens = build_chat_tokens(&self.tokenizer, user_input);

        // Prefill
        for &tok in &prompt_tokens[..prompt_tokens.len() - 1] {
            self.model.forward(&mut self.gpu, tok);
        }

        let decode_start = std::time::Instant::now();
        let mut generated_ids: Vec<u32> = Vec::new();
        let mut token =
            self.model
                .forward(&mut self.gpu, prompt_tokens[prompt_tokens.len() - 1]);
        let mut interrupted = false;
        let mut think_count = 0u32;
        let mut response_count = 0u32;
        let max_think = 256u32;
        let mut in_think = false;
        let mut think_forced = false;
        let mut think_done = false;

        let think_end_tokens: Vec<u32> = self
            .tokenizer
            .encode("</think>\n", false)
            .unwrap()
            .get_ids()
            .to_vec();

        for gen_count in 0..self.max_tokens + max_think {
            if token == self.im_end_id || token == self.eos_id {
                break;
            }
            if let Some(flag) = cancel {
                if flag.load(Ordering::Relaxed) {
                    interrupted = true;
                    break;
                }
            }
            generated_ids.push(token);

            if !think_forced {
                if gen_count < 5 {
                    let partial = self
                        .tokenizer
                        .decode(&generated_ids, true)
                        .unwrap_or_default();
                    if partial.contains("<think>") {
                        in_think = true;
                    }
                }
                if in_think {
                    think_count += 1;
                    if think_count % 20 == 0 {
                        let partial = self
                            .tokenizer
                            .decode(&generated_ids, true)
                            .unwrap_or_default();
                        if partial.contains("</think>") {
                            in_think = false;
                        }
                    }
                    if think_count >= max_think {
                        for &t in &think_end_tokens {
                            generated_ids.push(t);
                            token = self.model.forward(&mut self.gpu, t);
                        }
                        in_think = false;
                        think_forced = true;
                        continue;
                    }
                }
            }

            if !in_think && (think_done || think_forced) {
                response_count += 1;
                if response_count >= self.max_tokens {
                    break;
                }
            } else if !in_think && !think_done {
                response_count += 1;
                if response_count >= self.max_tokens {
                    break;
                }
            }

            if !in_think && think_count > 0 && !think_done {
                think_done = true;
            }

            token = self.model.forward(&mut self.gpu, token);
        }

        let decode_elapsed = decode_start.elapsed();
        let token_count = generated_ids.len();
        let tokens_per_sec = if token_count > 0 {
            token_count as f64 / decode_elapsed.as_secs_f64()
        } else {
            0.0
        };

        let full_text = self
            .tokenizer
            .decode(&generated_ids, true)
            .unwrap_or_default();

        // Strip <think>...</think> blocks
        let display = if let Some(pos) = full_text.rfind("</think>") {
            full_text[pos + 8..].trim()
        } else {
            full_text.trim_start_matches("<think>").trim()
        };
        let display = display
            .replace("</think>", "")
            .replace("<think>", "");
        let text = display.trim().to_string();

        self.conversation_log
            .push((user_input.to_string(), full_text));

        GenerateResult {
            text,
            token_count,
            tokens_per_sec,
            think_tokens: think_count,
            interrupted,
        }
    }

    /// Reset LoRA weights (forget all learned facts).
    pub fn forget(&mut self) {
        if self.model.lora.is_some() {
            self.model.lora =
                Some(LoraState::new(&self.gpu, LoraConfig::default(), &self.config));
            self.conversation_log.clear();
        }
    }

    /// Save LoRA adapter to safetensors file.
    pub fn save_lora(&mut self, path: &Path) {
        if let Some(ref lora) = self.model.lora {
            lora.save_safetensors(&mut self.gpu, path, &self.config);
        }
    }
}

// We need Clone for LoraConfig
impl Clone for LoraConfig {
    fn clone(&self) -> Self {
        Self {
            rank: self.rank,
            alpha: self.alpha,
            scale: self.scale,
            targets: self.targets,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_qa_pairs_basic() {
        let text = "Q: What is Zorbex?\nA: A purple mineral from Mars";
        let pairs = parse_qa_pairs(text);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, "What is Zorbex?");
        assert_eq!(pairs[0].1, "A purple mineral from Mars");
    }

    #[test]
    fn test_parse_qa_pairs_multiple() {
        let text = "Q: What is the dog's name?\nA: Rex\nQ: How old is Rex?\nA: 3 years old";
        let pairs = parse_qa_pairs(text);
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0], ("What is the dog's name?".into(), "Rex".into()));
        assert_eq!(pairs[1], ("How old is Rex?".into(), "3 years old".into()));
    }

    #[test]
    fn test_parse_qa_pairs_empty() {
        assert!(parse_qa_pairs("NONE").is_empty());
        assert!(parse_qa_pairs("").is_empty());
        assert!(parse_qa_pairs("Just some random text").is_empty());
    }

    #[test]
    fn test_parse_qa_pairs_missing_answer() {
        let text = "Q: What is this?\nSome other line";
        let pairs = parse_qa_pairs(text);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_parse_qa_pairs_no_space_after_colon() {
        let text = "Q:What is X?\nA:Something";
        let pairs = parse_qa_pairs(text);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, "What is X?");
        assert_eq!(pairs[0].1, "Something");
    }

    #[test]
    fn test_is_question_true() {
        assert!(is_question("What is the capital?"));
        assert!(is_question("Where is Wisconsin?"));
        assert!(is_question("Who is the president?"));
        assert!(is_question("When did it happen?"));
        assert!(is_question("Why is the sky blue?"));
        assert!(is_question("How does it work?"));
        assert!(is_question("Is there a park nearby?"));
        assert!(is_question("Do you know the answer?"));
        assert!(is_question("Can you help me?"));
        assert!(is_question("Tell me about dogs"));
        assert!(is_question("Is this a test?"));
    }

    #[test]
    fn test_is_question_false() {
        assert!(!is_question("My dog is named Rex"));
        assert!(!is_question("Zorbex is a purple mineral from Mars"));
        assert!(!is_question("The capital is Madison"));
        assert!(!is_question("I like cheese"));
    }

    // ── Integration tests requiring MODEL_DIR ─────────────────────

    fn make_test_session() -> Option<ChatSession> {
        let model_dir = match std::env::var("MODEL_DIR") {
            Ok(d) => d,
            Err(_) => return None,
        };
        Some(ChatSession::new(
            PathBuf::from(model_dir),
            64,   // low max_tokens for fast tests
            true, // learning enabled
            None,
        ))
    }

    #[test]
    #[ignore]
    fn test_teach_and_recall() {
        let mut session = make_test_session().expect("Set MODEL_DIR to run this test");

        // Teach a novel fact
        let result = session
            .teach("Q: What is Zorbex? A: Zorbex is a purple mineral from Mars", 5)
            .expect("teach should succeed");
        assert!(result.loss > 0.0);
        assert!(result.loss < 10.0);

        // Ask about it
        let response = session.generate("What is Zorbex?", None);
        let lower = response.text.to_lowercase();
        assert!(
            lower.contains("purple") || lower.contains("mineral") || lower.contains("mars"),
            "Expected recall of taught fact, got: {}",
            response.text
        );
    }

    #[test]
    #[ignore]
    fn test_teach_does_not_forget_anchors() {
        let mut session = make_test_session().expect("Set MODEL_DIR to run this test");

        // Teach a new fact
        session.teach("Q: What is Glorpium? A: A shiny crystal found in caves", 5);

        // Ask an anchor question
        let response = session.generate("What is the capital of Wisconsin?", None);
        let lower = response.text.to_lowercase();
        assert!(
            lower.contains("madison"),
            "Expected anchor recall of Madison, got: {}",
            response.text
        );
    }

    #[test]
    #[ignore]
    fn test_forget_resets() {
        let mut session = make_test_session().expect("Set MODEL_DIR to run this test");

        // Teach, then forget
        session.teach("Q: What is Blipnox? A: A rare gemstone from Neptune", 5);
        session.forget();

        // Should not recall the taught fact
        let response = session.generate("What is Blipnox?", None);
        let lower = response.text.to_lowercase();
        // After forget, unlikely to mention the specific taught answer
        // (though the model might hallucinate — we just check the teach path resets)
        assert!(
            !lower.contains("neptune") || !lower.contains("gemstone"),
            "Expected forget to clear taught fact, but it was recalled: {}",
            response.text
        );
    }

    #[test]
    #[ignore]
    fn test_save_and_load_lora() {
        let model_dir =
            std::env::var("MODEL_DIR").expect("Set MODEL_DIR to run this test");
        let tmp_path = std::env::temp_dir().join("test_lora_checkpoint.safetensors");

        // Session 1: teach and save
        {
            let mut session = ChatSession::new(
                PathBuf::from(&model_dir),
                64,
                true,
                None,
            );
            session.teach("Q: What is Quazzle? A: A fizzy drink from Saturn", 5);
            session.save_lora(&tmp_path);
        }

        // Session 2: load and recall
        {
            let mut session = ChatSession::new(
                PathBuf::from(&model_dir),
                64,
                false, // no learning, just inference
                Some(tmp_path.clone()),
            );
            let response = session.generate("What is Quazzle?", None);
            let lower = response.text.to_lowercase();
            assert!(
                lower.contains("fizzy") || lower.contains("drink") || lower.contains("saturn"),
                "Expected recall from loaded LoRA, got: {}",
                response.text
            );
        }

        // Cleanup
        let _ = std::fs::remove_file(&tmp_path);
    }
}
