use std::path::PathBuf;

use tensorbend_rs::gpu;
use tensorbend_rs::model;
use tensorbend_rs::weights;

fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model_dir> [max_tokens]", args[0]);
        eprintln!();
        eprintln!("  model_dir: path to directory containing safetensors + config.json");
        eprintln!("  max_tokens: number of tokens to generate (default: 32)");
        std::process::exit(1);
    }

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

    // Simple greedy generation from BOS token
    // Qwen uses token 151643 as BOS typically, but this depends on the tokenizer
    let bos_token: u32 = 151643;
    let mut token = bos_token;

    println!("Generating {} tokens (BOS={})", max_tokens, bos_token);

    let start = std::time::Instant::now();
    for _i in 0..max_tokens {
        let next_token = model.forward(&mut gpu_ctx, token);

        print!("[{}] ", next_token);
        token = next_token;

        // EOS check (Qwen: 151645)
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
