//! Interactive chat with JIT LoRA learning.
//!
//! Usage:
//!   cargo run --release --example chat -- <model_dir> [--max-tokens 256] [--learn] [--lora <path>]
//!
//! With --learn, the model trains LoRA adapters on each conversation turn
//! so it remembers facts you teach it.
//!
//! With --lora, loads a pre-trained LoRA adapter from a safetensors file.
//! Combine with --learn to continue training from a checkpoint.

use std::io::{self, Read as _, Write};
use std::os::unix::io::AsRawFd;
use std::path::PathBuf;
use std::time::Instant;

use tokenizers::Tokenizer;

use tensorbend_rs::gpu::GpuContext;
use tensorbend_rs::lora::{LoraConfig, LoraState};
use tensorbend_rs::model::{CacheSlot, Model};
use tensorbend_rs::train::{EwcConfig, TrainState};
use tensorbend_rs::weights;

/// Build Qwen chat-template token sequence for a single user turn.
fn build_chat_tokens(tokenizer: &Tokenizer, user_text: &str) -> Vec<u32> {
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
fn build_train_tokens(
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
/// Only extracts from user messages — never from AI responses (which may hallucinate).
fn build_extract_prompt(tokenizer: &Tokenizer, user_msg: &str) -> Vec<u32> {
    let im_start = tokenizer.token_to_id("<|im_start|>").unwrap();
    let im_end = tokenizer.token_to_id("<|im_end|>").unwrap();
    let nl = tokenizer.encode("\n", false).unwrap();
    let nl_ids: Vec<u32> = nl.get_ids().to_vec();

    let system_text = "Extract facts as Q/A pairs. Copy the user's exact words for answers. Do not verify or change facts.";

    let prompt_text = format!(
        "Text: Blobly is a waterpark in Wisconsin.\nQ: What is Blobly?\nA: Blobly is a waterpark in Wisconsin.\n\nText: Jenny's dog is named Rex and he is 3 years old.\nQ: What is Jenny's dog's name?\nA: Rex\nQ: How old is Rex?\nA: 3 years old\n\nText: How's the weather today?\nNONE\n\nText: {user_msg}\n"
    );

    let sys_enc = tokenizer.encode(format!("system\n{system_text}"), false).unwrap();
    let usr_enc = tokenizer.encode(format!("user\n{prompt_text}"), false).unwrap();
    // Prefill assistant with "Q:" to force it to start with a question (skip <think>)
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
fn parse_qa_pairs(text: &str) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim();
        if let Some(q) = line.strip_prefix("Q: ").or_else(|| line.strip_prefix("Q:")) {
            let q = q.trim().to_string();
            if i + 1 < lines.len() {
                let next = lines[i + 1].trim();
                if let Some(a) = next.strip_prefix("A: ").or_else(|| next.strip_prefix("A:")) {
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

/// Run extraction on a secondary cache slot (no LoRA, base model only).
/// Returns parsed Q/A pairs.
fn extract_qa(
    gpu: &mut GpuContext,
    model: &mut Model,
    slot: &mut CacheSlot,
    tokenizer: &Tokenizer,
    user_msg: &str,
    im_end_id: u32,
    eos_id: u32,
) -> Vec<(String, String)> {
    let prompt = build_extract_prompt(tokenizer, user_msg);

    // Swap to extraction cache, disable LoRA
    let saved_tokens = std::mem::take(&mut model.generated_tokens);
    model.swap_cache(slot);
    let lora = model.lora.take();
    model.seq_len = 0;

    // Prefill all but last token
    for &tok in &prompt[..prompt.len() - 1] {
        model.forward(gpu, tok);
    }

    // Generate extraction response — track per-token confidence
    let mut ids: Vec<u32> = Vec::new();
    let mut token = model.forward(gpu, prompt[prompt.len() - 1]);
    let mut prob_sum = 0.0f32;
    let mut prob_count = 0u32;

    for _ in 0..128 {
        if token == im_end_id || token == eos_id {
            break;
        }
        ids.push(token);
        prob_sum += model.last_token_prob;
        prob_count += 1;
        token = model.forward(gpu, token);
    }

    let avg_confidence = if prob_count > 0 { prob_sum / prob_count as f32 } else { 0.0 };

    // Restore: swap cache back, re-enable LoRA
    model.lora = lora;
    model.swap_cache(slot);
    model.generated_tokens = saved_tokens;

    let raw_text = tokenizer.decode(&ids, true).unwrap_or_default();
    let full_text = format!("Q:{}", raw_text);
    eprintln!("\x1b[90m[extract prompt: {:?}]\x1b[0m", user_msg);
    eprintln!("\x1b[90m[extract full ({} tok, conf={:.2}): {:?}]\x1b[0m",
        ids.len(), avg_confidence, &full_text[..full_text.len().min(300)]);

    // Low confidence = likely hallucination, skip
    if avg_confidence < 0.3 {
        eprintln!("\x1b[90m[extract: low confidence {:.2}, skipping]\x1b[0m", avg_confidence);
        return Vec::new();
    }

    parse_qa_pairs(&full_text)
}

/// Set stdin to raw non-blocking mode, returns the old termios to restore later.
fn set_raw_nonblocking() -> libc::termios {
    unsafe {
        let fd = io::stdin().as_raw_fd();
        let mut old: libc::termios = std::mem::zeroed();
        libc::tcgetattr(fd, &mut old);
        let mut raw = old;
        raw.c_lflag &= !(libc::ICANON | libc::ECHO);
        raw.c_cc[libc::VMIN] = 0;
        raw.c_cc[libc::VTIME] = 0;
        libc::tcsetattr(fd, libc::TCSANOW, &raw);
        // Set non-blocking
        let flags = libc::fcntl(fd, libc::F_GETFL);
        libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK);
        old
    }
}

/// Restore terminal to previous mode.
fn restore_terminal(old: &libc::termios) {
    unsafe {
        let fd = io::stdin().as_raw_fd();
        libc::tcsetattr(fd, libc::TCSANOW, old);
        let flags = libc::fcntl(fd, libc::F_GETFL);
        libc::fcntl(fd, libc::F_SETFL, flags & !libc::O_NONBLOCK);
    }
}

/// Check if ESC was pressed (non-blocking).
fn esc_pressed() -> bool {
    let mut buf = [0u8; 8];
    match io::stdin().read(&mut buf) {
        Ok(n) if n > 0 => buf[..n].contains(&0x1B), // ESC = 0x1B
        _ => false,
    }
}

fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model_dir> [--max-tokens N] [--learn] [--lora <path>]", args[0]);
        eprintln!();
        eprintln!("  model_dir    Directory with safetensors, config.json, tokenizer.json");
        eprintln!("  --max-tokens Maximum response tokens (default: 256)");
        eprintln!("  --learn      Enable JIT LoRA learning from conversation");
        eprintln!("  --lora PATH  Load LoRA adapter from safetensors file");
        std::process::exit(1);
    }

    let model_dir = PathBuf::from(&args[1]);
    let mut max_tokens: u32 = 256;
    let mut learning_enabled = false;
    let mut lora_path: Option<PathBuf> = None;
    {
        let mut i = 2;
        while i < args.len() {
            match args[i].as_str() {
                "--max-tokens" if i + 1 < args.len() => {
                    max_tokens = args[i + 1].parse().unwrap_or(256);
                    i += 2;
                }
                "--learn" => {
                    learning_enabled = true;
                    i += 1;
                }
                "--lora" if i + 1 < args.len() => {
                    lora_path = Some(PathBuf::from(&args[i + 1]));
                    i += 2;
                }
                _ => i += 1,
            }
        }
    }
    let max_seq_len: u32 = 2048;

    // ── Load tokenizer ──
    let tok_path = model_dir.join("tokenizer.json");
    eprintln!("Loading tokenizer from {:?}", tok_path);
    let tokenizer = Tokenizer::from_file(&tok_path).expect("failed to load tokenizer.json");

    let im_end_id = tokenizer.token_to_id("<|im_end|>").unwrap_or(151645);
    let eos_id = tokenizer.token_to_id("<|endoftext|>").unwrap_or(151643);

    // ── Load model ──
    eprintln!("Loading model from {:?}...", model_dir);
    let load_start = Instant::now();

    let config = weights::ModelConfig::from_file(&model_dir.join("config.json"));
    let quant_config = weights::QuantConfig::from_file(&model_dir.join("quantize_config.json"));

    eprintln!(
        "  {} layers, hidden={}, heads={}/{}kv, head_dim={}, vocab={}",
        config.num_hidden_layers,
        config.hidden_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.vocab_size,
    );

    let mut gpu = GpuContext::new();
    let (model_weights, raw_norms) = weights::load_weights(&gpu, &model_dir, &config);
    let mut model = Model::new(&gpu, config.clone(), quant_config, model_weights, max_seq_len);

    for (i, norm_data) in raw_norms.layers.iter().enumerate() {
        if let Some((q_bytes, k_bytes)) = norm_data {
            model.init_qknorm_params(&gpu, i, q_bytes, k_bytes);
        }
    }

    // ── Initialize LoRA ──
    let lora_config = LoraConfig::default();
    if learning_enabled || lora_path.is_some() {
        if let Some(ref path) = lora_path {
            eprintln!("Loading LoRA adapter from {:?}", path);
            let lora = LoraState::load_safetensors(&gpu, path, &config);
            model.lora = Some(lora);
        } else {
            eprintln!("JIT LoRA enabled (rank={}, alpha={})", lora_config.rank, lora_config.alpha);
            let lora = LoraState::new(&gpu, lora_config, &config);
            model.lora = Some(lora);
        }
    }

    // ── Initialize training state ──
    let mut train_state = if learning_enabled {
        let mut ts = TrainState::new(&gpu, &config, &LoraConfig::default());
        // Enable EWC anti-forgetting by default
        ts.enable_ewc(&gpu, EwcConfig::default(), &config, &LoraConfig::default());
        eprintln!("EWC anti-forgetting enabled (λ={}, refresh every {} steps)",
            EwcConfig::default().lambda, EwcConfig::default().refresh_interval);
        Some(ts)
    } else {
        None
    };

    // ── Allocate extraction cache slot (second KV cache on GPU) ──
    let mut extract_slot = if learning_enabled {
        let slot = model.alloc_cache_slot(&gpu, max_seq_len);
        eprintln!("Extraction cache allocated (second KV cache on GPU)");
        Some(slot)
    } else {
        None
    };

    let load_elapsed = load_start.elapsed();
    eprintln!("Model loaded in {:.2}s", load_elapsed.as_secs_f64());
    if learning_enabled {
        eprintln!("Learning mode: ON — facts extracted automatically from conversation");
    }
    eprintln!();
    eprintln!("Commands:");
    eprintln!("  /teach <fact>              Learn a plain fact");
    eprintln!("  /teach Q: <question> A: <answer>  Learn a Q&A pair");
    eprintln!("  /forget                    Reset LoRA weights");
    eprintln!("  /save [path]               Checkpoint adapter to safetensors");
    eprintln!("  /quit                      Exit");
    eprintln!();

    // ── Anchor replay buffer — pre-tokenized facts for anti-forgetting ──
    let anchor_qa: Vec<(&str, &str)> = vec![
        ("What is the capital of Wisconsin?", "Madison"),
        ("What is Wisconsin known for?", "Dairy farming and cheese production"),
        ("What is the largest city in Wisconsin?", "Milwaukee"),
        ("What are the Wisconsin Dells?", "A popular waterpark and resort area in Wisconsin"),
        ("What Great Lakes border Wisconsin?", "Lake Michigan and Lake Superior"),
        ("What is the state animal of Wisconsin?", "The badger"),
        ("What is a famous Wisconsin food?", "Cheese curds"),
    ];
    let mut anchor_tokens: Vec<Vec<u32>> = anchor_qa.iter()
        .map(|(q, a)| build_train_tokens(&tokenizer, q, a))
        .collect();
    let mut anchor_rng = 42u64; // simple RNG for picking random anchors

    // Seed with anchor facts or load cached anchor
    let anchor_path = std::path::Path::new("anchor.lora.safetensors");
    if let Some(ref mut ts) = train_state {
        if anchor_path.exists() {
            eprintln!("\x1b[90m[Loading anchor from {:?}]\x1b[0m", anchor_path);
            model.lora = Some(LoraState::load_safetensors(&gpu, anchor_path, &config));
        } else {
            let seed_start = Instant::now();
            for (i, toks) in anchor_tokens.iter().enumerate() {
                eprint!("\r\x1b[90m[Seeding anchor {}/{} ...]\x1b[0m", i + 1, anchor_tokens.len());
                io::stderr().flush().unwrap();
                let _loss = ts.train_on_tokens(&mut gpu, &mut model, toks, 1);
            }
            let seed_elapsed = seed_start.elapsed();

            if let Some(ref lora) = model.lora {
                lora.save_safetensors(&mut gpu, anchor_path, &config);
                eprintln!("\r\x1b[90m[Seeded {} facts in {:.1}s, saved to {:?}]\x1b[0m",
                    anchor_tokens.len(), seed_elapsed.as_secs_f64(), anchor_path);
            }
        }
        ts.snapshot_anchor_grads(&mut gpu, &config, &LoraConfig::default());
        eprintln!("\x1b[90m[EWC anchored, {} replay facts]\x1b[0m", anchor_tokens.len());
    }

    // ── Conversation history for training ──
    let mut conversation_log: Vec<(String, String)> = Vec::new(); // (user, assistant)

    loop {
        model.seq_len = 0;
        model.generated_tokens.clear();

        eprint!("\x1b[1mYou:\x1b[0m ");
        io::stderr().flush().unwrap();

        let mut user_input = String::new();
        if io::stdin().read_line(&mut user_input).unwrap() == 0 {
            break;
        }
        let user_input = user_input.trim().to_string();
        if user_input.is_empty() {
            continue;
        }

        // ── Commands ──
        if user_input == "/quit" || user_input == "/exit" {
            break;
        }
        if user_input == "/forget" {
            if model.lora.is_some() {
                eprintln!("\x1b[90m[LoRA reset — forgetting all learned facts]\x1b[0m");
                model.lora = Some(LoraState::new(&gpu, LoraConfig::default(), &config));
                conversation_log.clear();
            }
            continue;
        }
        if user_input.starts_with("/save") {
            let save_path = user_input
                .strip_prefix("/save")
                .unwrap()
                .trim();
            let save_path = if save_path.is_empty() {
                "lora_adapter.safetensors"
            } else {
                save_path
            };
            if let Some(ref lora) = model.lora {
                let path = std::path::Path::new(save_path);
                lora.save_safetensors(&mut gpu, path, &config);
                eprintln!("\x1b[90m[Saved LoRA adapter to {:?}]\x1b[0m", path);
            } else {
                eprintln!("No LoRA adapter to save.");
            }
            continue;
        }
        if user_input.starts_with("/teach ") {
            let body = user_input.strip_prefix("/teach ").unwrap().trim();
            if let Some(ref mut ts) = train_state {
                // Support "Q: ... A: ..." format or plain fact
                let (question, answer) = if let Some(qa) = body.split_once(" A: ") {
                    let q = qa.0.strip_prefix("Q: ").unwrap_or(qa.0);
                    (q.to_string(), qa.1.to_string())
                } else {
                    // Plain fact: train multiple Q&A framings
                    (format!("What do you know about: {body}"), body.to_string())
                };

                eprintln!("\x1b[90m[Learning: Q=\"{}\" A=\"{}\"]\x1b[0m", question, answer);

                let train_tokens = build_train_tokens(&tokenizer, &question, &answer);
                let num_epochs = 5;
                eprint!("\x1b[90m[Training");
                io::stderr().flush().unwrap();

                let train_start = Instant::now();
                let loss = ts.train_on_tokens(&mut gpu, &mut model, &train_tokens, num_epochs);
                let train_elapsed = train_start.elapsed();

                eprintln!(
                    " done: {} tok, {} epochs, loss={:.3}, {:.1}s, step={}]\x1b[0m",
                    train_tokens.len(),
                    num_epochs,
                    loss,
                    train_elapsed.as_secs_f64(),
                    ts.step,
                );

                // Snapshot anchor gradients after explicit teaching
                // so EWC will protect this knowledge during future training
                ts.snapshot_anchor_grads(&mut gpu, &config, &LoraConfig::default());
                eprintln!("\x1b[90m[EWC anchor updated]\x1b[0m");
            } else {
                eprintln!("Learning not enabled. Use --learn flag.");
            }
            continue;
        }

        // ── Normal chat ──
        let prompt_tokens = build_chat_tokens(&tokenizer, &user_input);
        let prompt_len = prompt_tokens.len();
        eprintln!("\x1b[90m[prompt: {} tokens]\x1b[0m", prompt_len);

        // ── Extract & train BEFORE generating response ──
        // Only extract from statements, not questions.
        let is_question = user_input.trim_end().ends_with('?')
            || user_input.to_lowercase().starts_with("what ")
            || user_input.to_lowercase().starts_with("where ")
            || user_input.to_lowercase().starts_with("who ")
            || user_input.to_lowercase().starts_with("when ")
            || user_input.to_lowercase().starts_with("why ")
            || user_input.to_lowercase().starts_with("how ")
            || user_input.to_lowercase().starts_with("is there ")
            || user_input.to_lowercase().starts_with("do you ")
            || user_input.to_lowercase().starts_with("can you ")
            || user_input.to_lowercase().starts_with("tell me ");

        if let (Some(ref mut ts), Some(ref mut slot)) = (&mut train_state, &mut extract_slot) {
            if !is_question {
                let extract_start = Instant::now();
                let pairs = extract_qa(
                    &mut gpu, &mut model, slot, &tokenizer,
                    &user_input, im_end_id, eos_id,
                );
                let extract_elapsed = extract_start.elapsed();

                if pairs.is_empty() {
                    eprintln!(
                        "\x1b[90m[extract: no facts, {:.1}s]\x1b[0m",
                        extract_elapsed.as_secs_f64(),
                    );
                } else {
                    eprintln!(
                        "\x1b[90m[extract: {} fact(s) in {:.1}s]\x1b[0m",
                        pairs.len(),
                        extract_elapsed.as_secs_f64(),
                    );

                    // Train: 70% new, 30% replay (per jit-lora source)
                    // 3 epochs new + 1 epoch replay per fact
                    let train_start = Instant::now();
                    for (q, a) in &pairs {
                        eprintln!("\x1b[90m  Q: {} → A: {}\x1b[0m", q, a);
                        eprint!("\x1b[90m  Training (3 new + 1 replay)");
                        io::stderr().flush().unwrap();

                        let new_tokens = build_train_tokens(&tokenizer, q, a);

                        // 3 epochs on new fact (~70%)
                        let loss = ts.train_on_tokens(&mut gpu, &mut model, &new_tokens, 3);

                        // 1 epoch on random replay (~30%)
                        anchor_rng ^= anchor_rng << 13;
                        anchor_rng ^= anchor_rng >> 7;
                        anchor_rng ^= anchor_rng << 17;
                        let anchor_idx = (anchor_rng as usize) % anchor_tokens.len();
                        let _anchor_loss = ts.train_on_tokens(
                            &mut gpu, &mut model, &anchor_tokens[anchor_idx], 1,
                        );

                        eprintln!(" loss={:.3}, step={}\x1b[0m", loss, ts.step);

                        // Add new fact to replay buffer
                        anchor_tokens.push(new_tokens);
                    }
                    let train_elapsed = train_start.elapsed();
                    eprintln!("\x1b[90m  [trained {:.1}s, replay buffer: {} facts]\x1b[0m",
                        train_elapsed.as_secs_f64(), anchor_tokens.len());

                    ts.snapshot_anchor_grads(&mut gpu, &config, &LoraConfig::default());
                }
            }
        }

        // Prefill all but last token
        let prefill_start = Instant::now();
        for &tok in &prompt_tokens[..prompt_tokens.len() - 1] {
            model.forward(&mut gpu, tok);
        }
        let prefill_elapsed = prefill_start.elapsed();
        eprintln!(
            "\x1b[90m[prefill: {:.2}s, {:.0} tok/s]\x1b[0m",
            prefill_elapsed.as_secs_f64(),
            prompt_len as f64 / prefill_elapsed.as_secs_f64(),
        );

        // Decode starting from last prompt token
        print!("\x1b[1mAI:\x1b[0m ");
        io::stdout().flush().unwrap();

        // Enable raw non-blocking stdin for ESC detection
        let old_term = set_raw_nonblocking();

        let decode_start = Instant::now();
        let mut generated_ids: Vec<u32> = Vec::new();
        let mut token = model.forward(&mut gpu, prompt_tokens[prompt_tokens.len() - 1]);
        let mut interrupted = false;
        let mut gen_count = 0u32;
        let mut think_count = 0u32;
        let mut response_count = 0u32;
        let max_think = 256u32;
        let mut in_think = false;
        let mut think_forced = false;
        let mut think_done = false;
        let mut last_status = Instant::now();

        // Encode </think>\n once for reuse
        let think_end_tokens: Vec<u32> = tokenizer.encode("</think>\n", false)
            .unwrap().get_ids().to_vec();

        for _ in 0..max_tokens + max_think {
            if token == im_end_id || token == eos_id {
                break;
            }
            if esc_pressed() {
                interrupted = true;
                break;
            }
            generated_ids.push(token);
            gen_count += 1;

            // Track think state cheaply — only decode to check transitions
            if !think_forced {
                if gen_count <= 5 {
                    // Only check for <think> in first few tokens
                    let partial = tokenizer.decode(&generated_ids, true).unwrap_or_default();
                    if partial.contains("<think>") {
                        in_think = true;
                    }
                }
                if in_think {
                    think_count += 1;
                    // Check for natural end periodically
                    if think_count % 20 == 0 {
                        let partial = tokenizer.decode(&generated_ids, true).unwrap_or_default();
                        if partial.contains("</think>") {
                            in_think = false;
                        }
                    }
                    // Force end if over budget
                    if think_count >= max_think {
                        for &t in &think_end_tokens {
                            generated_ids.push(t);
                            token = model.forward(&mut gpu, t);
                        }
                        in_think = false;
                        think_forced = true;
                        // token is now set to continue after </think>
                        continue;
                    }
                }
            }

            // Count response tokens (after think ends)
            if !in_think && (think_done || think_forced) {
                response_count += 1;
                if response_count >= max_tokens {
                    break;
                }
            } else if !in_think && !think_done {
                // Never entered think — count all as response
                response_count += 1;
                if response_count >= max_tokens {
                    break;
                }
            }

            // Mark think as done when it naturally ends
            if in_think {
                // (handled above)
            } else if think_count > 0 && !think_done {
                think_done = true;
            }

            // Show progress
            if last_status.elapsed().as_millis() >= 200 {
                if in_think {
                    eprint!("\r\x1b[90mThinking [{}/{}]\x1b[0m", think_count, max_think);
                } else {
                    eprint!("\r\x1b[90m[generating {}/{}]\x1b[0m", response_count, max_tokens);
                }
                io::stderr().flush().unwrap();
                last_status = Instant::now();
            }

            token = model.forward(&mut gpu, token);
        }

        // Clear status line, restore terminal
        eprint!("\r\x1b[K");
        restore_terminal(&old_term);

        if interrupted {
            eprintln!("\x1b[90m[interrupted at {} tokens]\x1b[0m\n", gen_count);
            continue;
        }

        // Decode and strip <think>...</think> blocks
        let full_text = tokenizer.decode(&generated_ids, true).unwrap_or_default();

        // Log think content if present
        if let Some(start) = full_text.find("<think>") {
            let think_end = full_text.find("</think>").unwrap_or(full_text.len());
            let think_content = full_text[start + 7..think_end].trim();
            if !think_content.is_empty() {
                eprintln!("\x1b[90m[think ({} tok): {}]\x1b[0m",
                    think_count, &think_content[..think_content.len().min(500)]);
            }
        } else if think_count > 0 {
            eprintln!("\x1b[90m[think ({} tok, no tags found in output)]\x1b[0m", think_count);
        }

        let display = if let Some(pos) = full_text.rfind("</think>") {
            full_text[pos + 8..].trim()
        } else {
            full_text.trim_start_matches("<think>").trim()
        };
        let display = display.replace("</think>", "").replace("<think>", "");
        print!("{}", display.trim());

        let decode_elapsed = decode_start.elapsed();
        let gen_count = generated_ids.len();
        let decode_tps = if gen_count > 0 {
            gen_count as f64 / decode_elapsed.as_secs_f64()
        } else {
            0.0
        };

        let response_text = tokenizer.decode(&generated_ids, true).unwrap_or_default();

        println!();
        eprintln!(
            "\x1b[90m[decode: {} tok in {:.2}s = {:.1} tok/s]\x1b[0m",
            gen_count,
            decode_elapsed.as_secs_f64(),
            decode_tps,
        );

        if let Some(ref mut _ts) = train_state {
            conversation_log.push((user_input.clone(), response_text.clone()));
        }
        eprintln!();
    }

    eprintln!("Bye!");
}
