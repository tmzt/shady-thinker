use std::path::PathBuf;

use shady_thinker::inference::InferenceSession;

/// Single source of truth: `common/src/fast_thinker_system_prompt.txt`
const FAST_THINKER_SYSTEM_PROMPT: &str =
    include_str!("../../../common/src/fast_thinker_system_prompt.txt");

fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();

    // Dispatch subcommands
    if args.len() >= 2 && args[1] == "cache-prefix" {
        cmd_cache_prefix(&args);
        return;
    }

    if args.len() >= 2 && args[1] == "query" {
        cmd_query(&args);
        return;
    }

    if args.len() < 2 {
        eprintln!("Usage:");
        eprintln!("  {} <model_dir> [max_tokens]", args[0]);
        eprintln!("  {} cache-prefix <model_dir>", args[0]);
        eprintln!("  {} query <model_dir> \"<user query>\" [max_tokens]", args[0]);
        eprintln!();
        eprintln!("  model_dir:  path to directory containing safetensors + config.json");
        eprintln!("  max_tokens: number of tokens to generate (default: 32)");
        std::process::exit(1);
    }

    cmd_generate(&args);
}

/// `query <model_dir> "<user query>" [max_tokens]`:
/// Run the fast thinker session with system prompt prefix and a user query.
/// Uses the incremental prefill path if a prefix cache exists.
fn cmd_query(args: &[String]) {
    if args.len() < 4 {
        eprintln!("Usage: {} query <model_dir> \"<user query>\" [max_tokens] [--full]", args[0]);
        std::process::exit(1);
    }

    let model_dir = std::path::PathBuf::from(&args[2]);
    let user_query = &args[3];
    let max_tokens: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(64);
    let force_full = args.iter().any(|s| s == "--full");
    // --kv-only: bypass prefill_gptq, use token-by-token path to verify numerical correctness
    let kv_only = args.iter().any(|s| s == "--kv-only");
    // --serial: use token-by-token path for the full prompt (generate_tokens_thinking with no think_config)
    let serial = args.iter().any(|s| s == "--serial");
    // --debug-normed: run prefill, readback normed output, compare batch vs serial
    let debug_normed = args.iter().any(|s| s == "--debug-normed");
    const MAX_SEQ_LEN: u32 = 4096;

    log::info!("[query] loading model from {:?} (full_prefill={})", model_dir, force_full);
    let mut session = InferenceSession::new(model_dir.clone(), MAX_SEQ_LEN);

    let tokenizer = common::tokenizer::Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .expect("failed to load tokenizer.json");

    // Build full prompt: encode system and user turns SEPARATELY to avoid BPE merging
    // across the boundary. The prefix cache was built from tokenize(system_text) alone,
    // so input_ids[prefix_len..] must be tokenize(user_turn) — not a slice of a joint encode.
    let system_text = format!("<|im_start|>system\n{FAST_THINKER_SYSTEM_PROMPT}<|im_end|>\n");
    let user_turn = format!("<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n");
    let system_ids = tokenizer.encode(&system_text, false).expect("tokenize system");
    let user_ids = tokenizer.encode(&user_turn, false).expect("tokenize user turn");
    let mut input_ids = system_ids;
    input_ids.extend_from_slice(&user_ids);

    if !force_full {
        // Set up system prompt prefix (try loading from cache first)
        let cache_path = InferenceSession::prefix_cache_path(&model_dir, &system_text);
        if !kv_only && session.try_load_prefix_cache(&cache_path) {
            log::info!("[query] prefix cache loaded (prefix_len={})", session.prefix_len);
        } else {
            if kv_only {
                // Force token-by-token prefill to bypass prefill_gptq batch path.
                // This is the bf16 code path (set bf16_mode=true temporarily).
                log::info!("[query] --kv-only: using token-by-token prefix build");
                session.model.bf16_mode = true;
                let ids = tokenizer.encode(&system_text, false).expect("tokenize system");
                session.set_prefix(&ids);
                session.model.bf16_mode = false; // restore for decode
            } else {
                log::info!("[query] no prefix cache — running set_prefix...");
                let ids = tokenizer.encode(&system_text, false).expect("tokenize system");
                session.set_prefix(&ids);
                session.save_prefix_cache(&cache_path).ok();
            }
            log::info!("[query] prefix set (prefix_len={})", session.prefix_len);
        }
    } else {
        log::info!("[query] --full: skipping prefix cache, using full prefill");
    }

    // EOS token IDs: 248046=<|im_end|>, 248044=<|endoftext|>
    // (We encode to confirm the actual ID rather than hardcoding.)
    let im_end_id = tokenizer.encode("<|im_end|>", false)
        .ok().and_then(|v| v.into_iter().next()).unwrap_or(248046);
    let eos_text_id = tokenizer.encode("<|endoftext|>", false)
        .ok().and_then(|v| v.into_iter().next()).unwrap_or(248044);
    let eos_ids = vec![im_end_id, eos_text_id];
    log::info!("[query] eos_ids: {:?}", eos_ids);

    // Enable JSON-constrained sampling (uses first-byte GPU gate + CPU EOS suppression)
    session.enable_json_mode(tokenizer.token_all_bytes(), eos_ids.clone());

    let n_query = input_ids.len() as u32 - session.prefix_len;
    log::info!("[query] full prompt: {} tokens (prefix={}, query+asst={})",
        input_ids.len(), session.prefix_len, n_query);
    if session.prefix_len > 0 {
        log::info!("[query] query token ids: {:?}", &input_ids[session.prefix_len as usize..]);
    }

    // --debug-n N: limit the prefill to first N tokens for bisection
    let debug_n: Option<usize> = args.windows(2)
        .find(|w| w[0] == "--debug-n")
        .and_then(|w| w[1].parse().ok());

    if debug_normed {
        // Run prefill only (from scratch, no prefix) and readback the normed output.
        // Use with --full (batch) or --full --serial to compare hidden states.
        eprintln!("[DEBUG] debug_normed=true, serial={}, debug_n={:?}", serial, debug_n);
        let h = session.model.config.hidden_size as u64;
        session.model.seq_len = 0;
        let use_ids: Vec<u32> = if let Some(n) = debug_n {
            input_ids[..n.min(input_ids.len())].to_vec()
        } else {
            input_ids.clone()
        };
        if serial {
            // Token-by-token — clear DeltaNet state to match the batch path's explicit zero-init.
            session.model.clear_deltanet_state(&mut session.gpu);
            for (i, &tok) in use_ids[..use_ids.len() - 1].iter().enumerate() {
                session.model.forward_kv_only(&mut session.gpu, tok);
                if (i + 1) % 4 == 0 { session.gpu.flush_and_wait(); }
            }
            session.gpu.flush_and_wait();
            session.model.forward(&mut session.gpu, *use_ids.last().unwrap());
        } else {
            session.model.prefill_gptq(&mut session.gpu, &use_ids);
            session.model.dispatch_lm_head(&mut session.gpu);
        }
        session.gpu.flush_and_wait();
        let normed_bytes = session.self.gpu.read_buffer(&session.model.state.normed, h * 4);
        let normed: Vec<f32> = normed_bytes.chunks(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        println!("[debug-normed n={}] seq={} first_16: {:?}", use_ids.len(), session.model.seq_len,
            &normed[..16.min(normed.len())]);
        let norm_sq: f32 = normed.iter().map(|x| x * x).sum::<f32>() / normed.len() as f32;
        println!("[debug-normed] rms={:.4}", norm_sq.sqrt());
        return;
    }

    let result = if serial {
        // Serial token-by-token path: inject_think=true but think_config=None → identical to normal
        // but bypasses prefill_gptq batch path. Use for numerical validation.
        log::info!("[query] --serial: using token-by-token generate_tokens_thinking");
        session.generate_tokens_thinking(&input_ids, max_tokens, &eos_ids, None)
    } else {
        session.generate_tokens(&input_ids, max_tokens, &eos_ids, None)
    };

    // Decode the output tokens
    let output = tokenizer.decode(&result.token_ids, true).unwrap_or_default();

    println!("\n=== OUTPUT ===\n{output}\n==============");
    println!("[{:.1} tok/s, {} tokens]", result.tokens_per_sec, result.token_count);
}

/// `cache-prefix <model_dir>`: pre-compute the KV cache for the system prompt
/// and save it alongside the model. Run this on macOS (fast Metal GPU) once;
/// the resulting file is loaded on Android at startup to skip the expensive
/// 2841-token prefill.
fn cmd_cache_prefix(args: &[String]) {
    if args.len() < 3 {
        eprintln!("Usage: {} cache-prefix <model_dir>", args[0]);
        std::process::exit(1);
    }

    let model_dir = PathBuf::from(&args[2]);
    const MAX_SEQ_LEN: u32 = 4096;

    log::info!("[cache-prefix] loading model from {:?}", model_dir);
    let mut session = InferenceSession::new(model_dir.clone(), MAX_SEQ_LEN);

    // Build the ChatML-formatted system prompt (same text used at runtime).
    // Format: <|im_start|>system\n{content}<|im_end|>\n
    let system_text = format!("<|im_start|>system\n{FAST_THINKER_SYSTEM_PROMPT}<|im_end|>\n");

    // Tokenize using the same common::tokenizer used by thinker_engine at runtime.
    let tokenizer = common::tokenizer::Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .expect("failed to load tokenizer.json");
    let ids: Vec<u32> = tokenizer.encode(&system_text, false)
        .expect("failed to tokenize system prompt");

    log::info!("[cache-prefix] system prompt: {} chars → {} tokens", system_text.len(), ids.len());

    // Run the prefill and capture the KV cache.
    session.set_prefix(&ids);

    // Determine cache file path (keyed by model shaders + prompt text).
    let cache_path = InferenceSession::prefix_cache_path(&model_dir, &system_text);
    log::info!("[cache-prefix] saving to {:?}", cache_path);

    session.save_prefix_cache(&cache_path).expect("failed to save prefix cache");

    println!("[cache-prefix] done: {:?}", cache_path);
}

/// Default subcommand: generate tokens from BOS (original debug/test mode).
fn cmd_generate(args: &[String]) {
    use shady_thinker::{gpu, model, weights};

    let model_dir = PathBuf::from(&args[1]);
    let max_tokens: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);
    let max_seq_len: u32 = 2048;

    // Load configs
    let config = weights::ModelConfig::from_file(&model_dir.join("config.json"));
    let quant_config = weights::QuantConfig::from_file(&model_dir.join("quantize_config.json"));

    log::info!("Model config: {:?}", config);
    log::info!("Quant config: {:?}", quant_config);
    log::info!(
        "Model: {} layers, hidden={}, heads={}/{}kv, head_dim={}, vocab={}",
        config.num_hidden_layers,
        config.hidden_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.vocab_size
    );

    // Initialize GPU
    let mut gpu_ctx = gpu::GpuContext::new();
    log::info!("GPU initialized");

    // Load weights
    let (model_weights, raw_norms) = weights::load_weights(&gpu_ctx, &model_dir, &config);
    log::info!("Weights loaded ({} layers)", config.num_hidden_layers);

    // Create model
    let mut model = model::Model::new(&gpu_ctx, config, quant_config, model_weights, max_seq_len);

    // Initialize per-layer QK norm uniform buffers with raw BF16 weight bytes
    for (i, norm_data) in raw_norms.layers.iter().enumerate() {
        if let Some((q_bytes, k_bytes)) = norm_data {
            model.init_qknorm_params(&gpu_ctx, i, q_bytes, k_bytes);
        }
    }
    log::info!("Model ready (attention path initialized)");

    let bos_token: u32 = 151643;
    let mut token = bos_token;

    println!("Generating {} tokens (BOS={})", max_tokens, bos_token);

    let start = std::time::Instant::now();
    for _i in 0..max_tokens {
        let next_token = model.forward(&mut gpu_ctx, token);

        print!("[{}] ", next_token);
        token = next_token;

        if next_token == 151645 {
            println!("\n<EOS>");
            break;
        }
    }

    let elapsed = start.elapsed();
    let tokens_per_sec = model.seq_len as f64 / elapsed.as_secs_f64();
    println!(
        "\n\n{} tokens in {:.2}s ({:.1} tok/s)",
        model.seq_len,
        elapsed.as_secs_f64(),
        tokens_per_sec
    );
}
