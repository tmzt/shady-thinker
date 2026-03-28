use std::path::Path;

fn run_encoder_test(model_name: &str) {
    let model_dir = Path::new("../../models").join(model_name);
    if !model_dir.exists() {
        eprintln!("SKIP: model dir {:?} not found", model_dir);
        return;
    }

    eprintln!("Loading ASR encoder from {:?}", model_dir);
    let t0 = std::time::Instant::now();
    let mut encoder = shady_thinker::asr_encoder::AsrEncoder::new(&model_dir);
    let load_ms = t0.elapsed().as_millis();
    eprintln!("Config: {:?} (loaded in {}ms)", encoder.config, load_ms);

    // Synthetic input: 10 tokens of d_model dimensions (as if from conv stem)
    let seq_len = 10u32;
    let d_model = encoder.config.d_model;
    let input: Vec<f32> = (0..seq_len * d_model).map(|i| (i as f32 * 0.001).sin()).collect();

    eprintln!("Running forward pass: seq_len={}, d_model={}", seq_len, d_model);
    let t1 = std::time::Instant::now();
    let output = encoder.forward(&input, seq_len);
    let fwd_ms = t1.elapsed().as_millis();

    let out_dim = encoder.config.output_dim;
    assert_eq!(output.len(), (seq_len * out_dim) as usize,
        "output size mismatch: expected {}x{}={}", seq_len, out_dim, seq_len * out_dim);

    let sum: f32 = output.iter().map(|x| x.abs()).sum();
    eprintln!("Output sum of abs: {:.4}, forward: {}ms", sum, fwd_ms);
    assert!(sum > 0.0, "output is all zeros");

    // Show first few values
    eprintln!("First 8 values: {:?}", &output[..8.min(output.len())]);

    eprintln!("PASS: {} ({} tokens × {} out_dim), load={}ms fwd={}ms",
        model_name, seq_len, out_dim, load_ms, fwd_ms);
}

#[test]
fn load_and_forward_0_6b() {
    let _ = env_logger::try_init();
    run_encoder_test("qwen3-asr-0.6b");
}

#[test]
fn load_and_forward_1_7b() {
    let _ = env_logger::try_init();
    run_encoder_test("qwen3-asr-1.7b");
}

/// Run with MODEL=qwen3-asr-1.7b to test a specific model.
#[test]
fn load_and_forward_env() {
    let _ = env_logger::try_init();
    if let Ok(model) = std::env::var("MODEL") {
        run_encoder_test(&model);
    } else {
        eprintln!("SKIP: set MODEL=<name> to run this test");
    }
}

/// Test GPU encoder output through the C offline decoder.
/// Run test_split_encoder first to generate /tmp/conv_stem_43_896.f32
#[test]
fn gpu_encoder_offline_decode() {
    let _ = env_logger::try_init();

    let stem_path = "/tmp/conv_stem_43_896.f32";
    if !std::path::Path::new(stem_path).exists() {
        eprintln!("SKIP: run test_split_encoder first");
        return;
    }

    let model_dir = std::path::Path::new("../../models/qwen3-asr-0.6b");
    if !model_dir.exists() { eprintln!("SKIP: model not found"); return; }

    // Load conv stem
    let stem_bytes = std::fs::read(stem_path).unwrap();
    let stem: &[f32] = bytemuck::cast_slice(&stem_bytes);
    let seq_len = 43u32;
    let d_model = 896u32;

    // GPU encoder
    let mut encoder = shady_thinker::asr_encoder::AsrEncoder::new(model_dir);
    let gpu_output = encoder.forward(stem, seq_len);
    let out_dim = encoder.config.output_dim;
    eprintln!("GPU encoder: {} tokens × {} = {} floats", seq_len, out_dim, gpu_output.len());
    eprintln!("  gpu[0][0:4]: {:?}", &gpu_output[..4]);

    // Write GPU output for C decoder test
    let gpu_path = "/tmp/gpu_enc_43_1024.f32";
    std::fs::write(gpu_path, bytemuck::cast_slice::<f32, u8>(&gpu_output)).unwrap();
    eprintln!("Wrote GPU encoder output to {gpu_path}");
    eprintln!("Run: /tmp/test_decode_gpu to test offline decode");
}

/// Compare GPU encoder output to C reference using real conv stem data.
/// Run test_split_encoder first to generate /tmp/conv_stem_*.f32 and /tmp/enc_ref_*.f32
#[test]
fn compare_to_c_reference() {
    let _ = env_logger::try_init();

    let stem_path = "/tmp/conv_stem_33_896.f32";
    let ref_path = "/tmp/enc_ref_33_1024.f32";
    if !std::path::Path::new(stem_path).exists() {
        eprintln!("SKIP: run test_split_encoder first to generate {}", stem_path);
        return;
    }

    let model_dir = std::path::Path::new("../../models/qwen3-asr-0.6b");
    if !model_dir.exists() { eprintln!("SKIP: model not found"); return; }

    // Load conv stem output (from C)
    let stem_bytes = std::fs::read(stem_path).unwrap();
    let stem: &[f32] = bytemuck::cast_slice(&stem_bytes);
    let seq_len = 33u32;
    let d_model = 896u32;
    assert_eq!(stem.len(), (seq_len * d_model) as usize);
    eprintln!("Loaded conv stem: {} tokens × {}", seq_len, d_model);
    eprintln!("  stem[0][0:4]: {:?}", &stem[..4]);

    // Load C reference encoder output
    let ref_bytes = std::fs::read(ref_path).unwrap();
    let reference: &[f32] = bytemuck::cast_slice(&ref_bytes);
    let out_dim = 1024u32;
    assert_eq!(reference.len(), (seq_len * out_dim) as usize);
    eprintln!("Loaded C reference: {} tokens × {}", seq_len, out_dim);
    eprintln!("  ref[0][0:4]: {:?}", &reference[..4]);

    // Run GPU encoder
    let mut encoder = shady_thinker::asr_encoder::AsrEncoder::new(model_dir);
    let t0 = std::time::Instant::now();
    let gpu_output = encoder.forward(stem, seq_len);
    let fwd_ms = t0.elapsed().as_millis();
    eprintln!("GPU forward: {}ms", fwd_ms);
    eprintln!("  gpu[0][0:4]: {:?}", &gpu_output[..4]);

    // Compare
    let mut max_diff: f32 = 0.0;
    let mut sum_diff: f32 = 0.0;
    for i in 0..reference.len() {
        let diff = (gpu_output[i] - reference[i]).abs();
        max_diff = max_diff.max(diff);
        sum_diff += diff;
    }
    let avg_diff = sum_diff / reference.len() as f32;
    eprintln!("Comparison: max_diff={:.6}, avg_diff={:.6}", max_diff, avg_diff);

    // bf16 weights have ~0.4% relative error, so allow some tolerance
    if max_diff > 0.5 {
        eprintln!("WARNING: max_diff > 0.5 — results may diverge");
    }
    eprintln!("DONE");
}

/// Test loading 1.7B decoder weights via both AsrDecoder and Model+bf16.
#[test]
fn load_decoder_1_7b() {
    let _ = env_logger::try_init();
    let model_dir = std::path::Path::new("../../models/qwen3-asr-1.7b");
    if !model_dir.exists() { eprintln!("SKIP: model not found"); return; }

    // Test AsrDecoder (standalone)
    let t0 = std::time::Instant::now();
    let decoder = shady_thinker::asr_decoder::AsrDecoder::new(model_dir, 2048);
    eprintln!("AsrDecoder config: {:?} ({}ms)", decoder.config, t0.elapsed().as_millis());

    // Test Model with bf16 weights (reuse existing infrastructure)
    let t1 = std::time::Instant::now();
    let gpu = shady_thinker::gpu::GpuContext::new();
    // ASR config has thinker_config.text_config nesting — extract it
    let config = {
        let raw: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(model_dir.join("config.json")).unwrap()
        ).unwrap();
        let text_cfg = &raw["thinker_config"]["text_config"];
        let cfg_str = serde_json::to_string(text_cfg).unwrap();
        let c: shady_thinker::weights::ModelConfig = serde_json::from_str(&cfg_str).unwrap();
        c
    };
    let quant_config = shady_thinker::weights::QuantConfig {
        bits: 16, group_size: 1, quant_method: "bf16".to_string(), sym: false,
    };
    let (weights, raw_norms) = shady_thinker::weights::load_weights_bf16(&gpu, model_dir, &config);
    let mut model = shady_thinker::model::Model::new(&gpu, config.clone(), quant_config, weights, 512);
    model.bf16_mode = true;
    for (i, norm) in raw_norms.layers.iter().enumerate() {
        if let Some((q, k)) = norm {
            model.init_qknorm_params(&gpu, i, q, k);
        }
    }
    eprintln!("Model+bf16 loaded ({}ms), config: {} layers, hidden={}",
        t1.elapsed().as_millis(), config.num_hidden_layers, config.hidden_size);

    // Test single forward step
    let t2 = std::time::Instant::now();
    let mut gpu = gpu;
    let token = model.forward(&mut gpu, 1); // token ID 1
    eprintln!("First token: {} ({}ms)", token, t2.elapsed().as_millis());
}

/// End-to-end: GPU encoder → GPU decoder on the 1.7B model.
/// Uses synthetic conv stem data, decodes with full ASR prompt structure.
#[test]
fn gpu_encoder_gpu_decoder_e2e() {
    let _ = env_logger::try_init();
    let model_dir = std::path::Path::new("../../models/qwen3-asr-1.7b");
    if !model_dir.exists() { eprintln!("SKIP: 1.7B model not found"); return; }

    // GPU encoder (1.7B)
    let mut encoder = shady_thinker::asr_encoder::AsrEncoder::new(model_dir);

    let seq_len = 33u32;
    let d_model = encoder.config.d_model;
    let input: Vec<f32> = (0..seq_len * d_model).map(|i| (i as f32 * 0.001).sin()).collect();

    let t0 = std::time::Instant::now();
    let enc_output = encoder.forward(&input, seq_len);
    eprintln!("Encoder: {} tokens × {} = {} floats in {}ms",
        seq_len, encoder.config.output_dim, enc_output.len(), t0.elapsed().as_millis());

    // GPU decoder (1.7B) — full ASR decode with prompt structure
    let (mut gpu, mut model) = shady_thinker::asr_decoder::load_bf16_model(model_dir, 512);

    let token_ids = shady_thinker::asr_decoder::gpu_asr_decode_tokens(
        &mut gpu, &mut model, &enc_output, seq_len);

    eprintln!("Decoded {} tokens: {:?}", token_ids.len(), &token_ids[..token_ids.len().min(20)]);
    assert!(!token_ids.is_empty(), "decoder produced no tokens");
    eprintln!("PASS: GPU encoder → GPU decoder e2e");
}

/// Test scaling with realistic sequence lengths.
/// 2s audio ≈ 25 tokens, 5s ≈ 62, 10s ≈ 125, 30s ≈ 375
#[test]
fn scaling_0_6b() {
    let _ = env_logger::try_init();
    let model_dir = std::path::Path::new("../../models/qwen3-asr-0.6b");
    if !model_dir.exists() { eprintln!("SKIP"); return; }

    let mut encoder = shady_thinker::asr_encoder::AsrEncoder::new(model_dir);
    let d = encoder.config.d_model;

    for &seq_len in &[25u32, 62, 125, 250, 375] {
        let input: Vec<f32> = (0..seq_len * d).map(|i| (i as f32 * 0.001).sin()).collect();
        let t0 = std::time::Instant::now();
        let output = encoder.forward(&input, seq_len);
        let ms = t0.elapsed().as_millis();
        let audio_s = seq_len as f32 * 0.08; // ~80ms per token
        let rtf = ms as f32 / (audio_s * 1000.0);
        eprintln!("seq_len={:>3} (~{:.0}s audio) → {}ms (RTF={:.2}x)",
            seq_len, audio_s, ms, rtf);
        assert_eq!(output.len(), (seq_len * encoder.config.output_dim) as usize);
    }
}

/// Real audio: GPU encoder + GPU decoder on 1.7B with actual conv stem data.
/// Run test_split_encoder first: /tmp/test_split_encoder ../../models/qwen3-asr-1.7b audio.f32 N
#[test]
fn gpu_full_pipeline_1_7b_real() {
    let _ = env_logger::try_init();

    let stem_path = "/tmp/conv_stem_17_1024.f32";
    if !std::path::Path::new(stem_path).exists() {
        eprintln!("SKIP: run test_split_encoder with 1.7B model first");
        return;
    }

    let model_dir = std::path::Path::new("../../models/qwen3-asr-1.7b");
    if !model_dir.exists() { eprintln!("SKIP: 1.7B model not found"); return; }

    // Load real conv stem (from C's conv2d)
    let stem_bytes = std::fs::read(stem_path).unwrap();
    let stem: &[f32] = bytemuck::cast_slice(&stem_bytes);
    let seq_len = 17u32;
    let d_model = 1024u32;
    assert_eq!(stem.len(), (seq_len * d_model) as usize);
    eprintln!("Loaded conv stem: {} tokens × {}", seq_len, d_model);

    // GPU encoder
    let mut encoder = shady_thinker::asr_encoder::AsrEncoder::new(model_dir);
    let t0 = std::time::Instant::now();
    let enc_output = encoder.forward(stem, seq_len);
    eprintln!("GPU encoder: {} tokens × {} in {}ms",
        seq_len, encoder.config.output_dim, t0.elapsed().as_millis());

    // Compare GPU encoder output to C reference
    let ref_path = "/tmp/enc_ref_17_2048.f32";
    if std::path::Path::new(ref_path).exists() {
        let ref_bytes = std::fs::read(ref_path).unwrap();
        let reference: &[f32] = bytemuck::cast_slice(&ref_bytes);
        assert_eq!(enc_output.len(), reference.len());
        let mut max_diff: f32 = 0.0;
        let mut sum_diff: f32 = 0.0;
        for i in 0..reference.len() {
            let diff = (enc_output[i] - reference[i]).abs();
            max_diff = max_diff.max(diff);
            sum_diff += diff;
        }
        eprintln!("Encoder vs C ref: max_diff={:.6}, avg_diff={:.6}",
            max_diff, sum_diff / reference.len() as f32);
        eprintln!("  gpu[0][0:4]: {:?}", &enc_output[..4]);
        eprintln!("  ref[0][0:4]: {:?}", &reference[..4]);
    }

    // GPU decoder — first test with C reference encoder output
    let (mut gpu, mut model) = shady_thinker::asr_decoder::load_bf16_model(model_dir, 512);

    // Try with C reference encoder output first to isolate encoder vs decoder
    let ref_path = "/tmp/enc_ref_17_2048.f32";
    let dec_input = if std::path::Path::new(ref_path).exists() {
        eprintln!("Using C reference encoder output for decoder test");
        let ref_bytes = std::fs::read(ref_path).unwrap();
        let reference: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&ref_bytes).to_vec();
        reference
    } else {
        enc_output.clone()
    };

    let t1 = std::time::Instant::now();
    let token_ids = shady_thinker::asr_decoder::gpu_asr_decode_tokens(
        &mut gpu, &mut model, &dec_input, seq_len);
    let total_ms = t1.elapsed().as_millis();

    eprintln!("GPU decoder: {} tokens in {}ms", token_ids.len(), total_ms);
    eprintln!("Token IDs: {:?}", &token_ids[..token_ids.len().min(30)]);

    assert!(!token_ids.is_empty(), "decoder produced no tokens");
    eprintln!("PASS: GPU full pipeline with real audio");
}

/// Debug: verify embedding + RMSNorm match Python reference.
#[test]
fn debug_decoder_verify_embedding() {
    let _ = env_logger::try_init();
    let model_dir = std::path::Path::new("../../models/qwen3-asr-1.7b");
    if !model_dir.exists() { eprintln!("SKIP"); return; }

    // Load Python reference
    let py_embed_path = "/tmp/py_embed_151644.f32";
    if !std::path::Path::new(py_embed_path).exists() {
        eprintln!("SKIP: run /tmp/verify_decoder.py first");
        return;
    }
    let py_embed_bytes = std::fs::read(py_embed_path).unwrap();
    let py_embed: &[f32] = bytemuck::cast_slice(&py_embed_bytes);
    let py_normed_bytes = std::fs::read("/tmp/py_normed_151644.f32").unwrap();
    let py_normed: &[f32] = bytemuck::cast_slice(&py_normed_bytes);

    eprintln!("Python embed[0:8]: {:?}", &py_embed[..8]);
    eprintln!("Python normed[0:8]: {:?}", &py_normed[..8]);

    // GPU embedding
    let (mut gpu, model) = shady_thinker::asr_decoder::load_bf16_model(model_dir, 512);

    // Run embedding shader
    let h = model.config.hidden_size;
    model.embedding(&mut gpu, 151644);
    gpu.flush();

    // Read back hidden state
    let hidden_bytes = gpu.read_buffer(&model.state.hidden, h as u64 * 4);
    let gpu_embed: &[f32] = bytemuck::cast_slice(&hidden_bytes);
    eprintln!("GPU   embed[0:8]: {:?}", &gpu_embed[..8]);

    // Compare embedding
    let mut max_diff: f32 = 0.0;
    for i in 0..h as usize {
        let diff = (gpu_embed[i] - py_embed[i]).abs();
        max_diff = max_diff.max(diff);
    }
    eprintln!("Embed max_diff: {:.8}", max_diff);
    assert!(max_diff < 0.001, "embedding mismatch: max_diff={}", max_diff);

    // Now check RMSNorm: embed → normed using layer 0 input_layernorm
    // RMSNorm with (1+w) scaling
    model.rmsnorm(&mut gpu, &model.state.hidden, &model.weights.layers[0].input_layernorm,
        &model.state.normed, h);
    gpu.flush();
    let normed_bytes = gpu.read_buffer(&model.state.normed, h as u64 * 4);
    let gpu_normed: &[f32] = bytemuck::cast_slice(&normed_bytes);
    eprintln!("GPU   normed[0:8]: {:?}", &gpu_normed[..8]);

    let mut norm_max_diff: f32 = 0.0;
    for i in 0..h as usize {
        let diff = (gpu_normed[i] - py_normed[i]).abs();
        norm_max_diff = norm_max_diff.max(diff);
    }
    eprintln!("Normed max_diff: {:.8}", norm_max_diff);

    // Q/K/V projections from normed
    let sa = model.weights.layers[0].self_attn().unwrap();
    let q_dim = model.config.num_attention_heads * model.config.head_dim; // 2048 (non-gated)
    let kv_dim = model.config.num_key_value_heads * model.config.head_dim; // 1024
    model.gptq_matvec(&mut gpu, "test_qproj",
        &model.state.normed, &sa.q_proj_qweight, &sa.q_proj_scales,
        &model.state.q_out, h, q_dim);
    model.gptq_matvec(&mut gpu, "test_kproj",
        &model.state.normed, &sa.k_proj_qweight, &sa.k_proj_scales,
        &model.state.k_out, h, kv_dim);
    model.gptq_matvec(&mut gpu, "test_vproj",
        &model.state.normed, &sa.v_proj_qweight, &sa.v_proj_scales,
        &model.state.v_out, h, kv_dim);
    gpu.flush();

    let q_bytes = gpu.read_buffer(&model.state.q_out, q_dim as u64 * 4);
    let gpu_q: &[f32] = bytemuck::cast_slice(&q_bytes);
    let k_bytes = gpu.read_buffer(&model.state.k_out, kv_dim as u64 * 4);
    let gpu_k: &[f32] = bytemuck::cast_slice(&k_bytes);
    let v_bytes = gpu.read_buffer(&model.state.v_out, kv_dim as u64 * 4);
    let gpu_v: &[f32] = bytemuck::cast_slice(&v_bytes);

    // Python: Q[-0.2485, 0.2878, -0.3837, 0.1787] K[0.981, 1.022, -1.447, -2.390] V[1.111, 1.022, 0.617, -4.170]
    eprintln!("GPU   Q[0:4]: {:?}", &gpu_q[..4]);
    eprintln!("GPU   K[0:4]: {:?}", &gpu_k[..4]);
    eprintln!("GPU   V[0:4]: {:?}", &gpu_v[..4]);

    // Run qknorm shader (Q/K norm + RoPE + KV cache write)
    model.fused_split_qknorm_kvstore(&mut gpu, 0);
    gpu.flush();

    let qproj_bytes = gpu.read_buffer(&model.state.q_proj, q_dim as u64 * 4);
    let gpu_qnormed: &[f32] = bytemuck::cast_slice(&qproj_bytes);
    let kout_bytes = gpu.read_buffer(&model.state.k_out, kv_dim as u64 * 4);
    let gpu_knormed: &[f32] = bytemuck::cast_slice(&kout_bytes);

    // Python: Q_normed[-0.9265, 0.3965, -0.1497, 0.3403] K_normed[1.3265, 2.4895, -4.8815, -4.2278]
    eprintln!("GPU   Q_normed[h0][0:4]: {:?}", &gpu_qnormed[..4]);
    eprintln!("GPU   K_normed[h0][0:4]: {:?}", &gpu_knormed[..4]);
}

/// Quick: does forward_argmax produce the same token as Python (65283)?
#[test]
fn verify_argmax_matches_python() {
    let _ = env_logger::try_init();
    let model_dir = std::path::Path::new("../../models/qwen3-asr-1.7b");
    if !model_dir.exists() { eprintln!("SKIP"); return; }
    let (mut gpu, mut model) = shady_thinker::asr_decoder::load_bf16_model(model_dir, 512);
    let token = model.forward_argmax(&mut gpu, 151644);
    eprintln!("GPU forward_argmax(151644) = {} (Python: 65283)", token);
    assert_eq!(token, 65283, "GPU decoder disagrees with Python reference");
}

/// Single token forward: compare GPU vs Python residuals per layer.
#[test]
fn verify_single_token_forward() {
    let _ = env_logger::try_init();
    let model_dir = std::path::Path::new("../../models/qwen3-asr-1.7b");
    if !model_dir.exists() { eprintln!("SKIP"); return; }
    let (mut gpu, mut model) = shady_thinker::asr_decoder::load_bf16_model(model_dir, 512);

    // Python residuals:
    // Layer 0: [1.859, -2.937, -13.263, -2.849]
    // Layer 1: [1.296, -7.269, -4.322, 2.129]
    // Layer 27: [4.609, 20.792, -0.067, 2.990]
    // Predicted: 65283

    // Manually run forward_argmax steps with instrumentation
    let h = model.config.hidden_size;
    model.embedding(&mut gpu, 151644);
    gpu.flush();
    gpu.copy_buffer(&model.state.hidden, &model.state.residual, h as u64 * 4);

    // Check residual matches embedding
    let res0 = gpu.read_buffer(&model.state.residual, h as u64 * 4);
    let r0: &[f32] = bytemuck::cast_slice(&res0);
    eprintln!("Initial residual[0:4]: {:?}", &r0[..4]);

    // Run layer 0
    let inter = model.config.intermediate_size;
    let nh = model.config.num_attention_heads;
    let nkv = model.config.num_key_value_heads;
    let hd = model.config.head_dim;
    let q_dim = nh * hd; // non-gated
    let kv_dim = nkv * hd;

    let layer = &model.weights.layers[0];
    model.rmsnorm(&mut gpu, &model.state.hidden, &layer.input_layernorm, &model.state.normed, h);

    let sa = layer.self_attn().unwrap();
    model.gptq_matvec(&mut gpu, "qproj_l0",
        &model.state.normed, &sa.q_proj_qweight, &sa.q_proj_scales,
        &model.state.q_out, h, q_dim);
    model.gptq_matvec(&mut gpu, "kproj_l0",
        &model.state.normed, &sa.k_proj_qweight, &sa.k_proj_scales,
        &model.state.k_out, h, kv_dim);
    model.gptq_matvec(&mut gpu, "vproj_l0",
        &model.state.normed, &sa.v_proj_qweight, &sa.v_proj_scales,
        &model.state.v_out, h, kv_dim);

    model.fused_split_qknorm_kvstore(&mut gpu, 0);
    model.gqa_attention(&mut gpu, 0);
    // No sigmoid_mul_gate (non-gated)
    model.gptq_matvec(&mut gpu, "oproj_l0",
        &model.state.attn_output, &sa.o_proj_qweight, &sa.o_proj_scales,
        &model.state.o_proj_out, nh * hd, h);

    // Post-attn: residual += o_proj_out, then norm
    model.add_rmsnorm(&mut gpu, &model.state.residual, &model.state.o_proj_out,
        &layer.post_attn_layernorm, &model.state.normed, h);

    // Check residual after attention
    gpu.flush();
    let res1 = gpu.read_buffer(&model.state.residual, h as u64 * 4);
    let r1: &[f32] = bytemuck::cast_slice(&res1);
    eprintln!("After attn L0 residual[0:4]: {:?}", &r1[..4]);
    // Python: after attn L0 (x + o): [1.522, 0.819, -2.574, -0.205]

    // Check normed (MLP input)
    gpu.flush();
    let normed0 = gpu.read_buffer(&model.state.normed, h as u64 * 4);
    let n0: &[f32] = bytemuck::cast_slice(&normed0);
    eprintln!("PostAttnNorm[0:4]: {:?}", &n0[..4]);
    // Python: [0.4453, 0.4192, -1.2779, -0.1052]

    // MLP
    model.gptq_matvec(&mut gpu, "gate_l0",
        &model.state.normed, &layer.gate_proj_qweight, &layer.gate_proj_scales,
        &model.state.gate_out, h, inter);
    model.gptq_matvec(&mut gpu, "up_l0",
        &model.state.normed, &layer.up_proj_qweight, &layer.up_proj_scales,
        &model.state.up_out, h, inter);
    gpu.flush();
    let gate0 = gpu.read_buffer(&model.state.gate_out, inter as u64 * 4);
    let g0: &[f32] = bytemuck::cast_slice(&gate0);
    let up0 = gpu.read_buffer(&model.state.up_out, inter as u64 * 4);
    let u0: &[f32] = bytemuck::cast_slice(&up0);
    eprintln!("Gate[0:4]: {:?}", &g0[..4]);
    eprintln!("Up[0:4]: {:?}", &u0[..4]);

    model.fused_silu_gptq_down(&mut gpu,
        &model.state.gate_out, &model.state.up_out,
        &layer.down_proj_qweight, &layer.down_proj_scales,
        &model.state.mlp_output, inter, h);

    // After MLP, residual gets updated at next layer's pre-attn
    // But let's check mlp_output first
    gpu.flush();
    let mlp0 = gpu.read_buffer(&model.state.mlp_output, h as u64 * 4);
    let m0: &[f32] = bytemuck::cast_slice(&mlp0);
    eprintln!("MLP L0 output[0:4]: {:?}", &m0[..4]);

    // Python: Layer 0 residual (embed + o + mlp): [1.859, -2.937, -13.263, -2.849]
    // So mlp_output should be ~[1.859-1.522, -2.937-0.819, -13.263+2.574, -2.849+0.205]
    //                        = [0.337, -3.756, -10.689, -2.644]
    eprintln!("Expected MLP: ~[0.337, -3.756, -10.689, -2.644]");

    model.seq_len += 1; // simulate the seq_len increment
    eprintln!("DONE");
}

/// Two-token verify: 151644→65283→?
#[test]
fn verify_two_token_forward() {
    let _ = env_logger::try_init();
    let model_dir = std::path::Path::new("../../models/qwen3-asr-1.7b");
    if !model_dir.exists() { eprintln!("SKIP"); return; }
    let (mut gpu, mut model) = shady_thinker::asr_decoder::load_bf16_model(model_dir, 512);
    let t1 = model.forward_argmax(&mut gpu, 151644);
    let t2 = model.forward_argmax(&mut gpu, t1);
    eprintln!("Token 0: 151644→{} (expected 65283)", t1);
    eprintln!("Token 1: {}→{} (expected 24)", t1, t2);
    assert_eq!(t1, 65283);
    assert_eq!(t2, 24);
}

/// Verify forward_embed_argmax matches forward_argmax for a token embedding.
#[test]
fn verify_embed_injection() {
    let _ = env_logger::try_init();
    let model_dir = std::path::Path::new("../../models/qwen3-asr-1.7b");
    if !model_dir.exists() { eprintln!("SKIP"); return; }

    // Get the token embedding for 151644
    let (mut gpu, mut model) = shady_thinker::asr_decoder::load_bf16_model(model_dir, 512);
    let h = model.config.hidden_size as usize;
    model.embedding(&mut gpu, 151644);
    gpu.flush();
    let embed_bytes = gpu.read_buffer(&model.state.hidden, h as u64 * 4);
    let embed: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&embed_bytes).to_vec();
    eprintln!("Embed[0:4]: {:?}", &embed[..4]);

    // Now use forward_embed_argmax with this embedding
    let t1 = model.forward_embed_argmax(&mut gpu, &embed);
    eprintln!("forward_embed_argmax(embed_151644) = {} (expected 65283)", t1);
    assert_eq!(t1, 65283, "embed injection doesn't match forward_argmax");
}

/// Verify all prefix tokens match Python argmax.
#[test]
fn verify_prefix_tokens() {
    let _ = env_logger::try_init();
    let model_dir = std::path::Path::new("../../models/qwen3-asr-1.7b");
    if !model_dir.exists() { eprintln!("SKIP"); return; }
    let (mut gpu, mut model) = shady_thinker::asr_decoder::load_bf16_model(model_dir, 512);
    let prefix = [151644u32, 8948, 198, 151645, 198, 151644, 872, 198, 151669];
    let expected = [65283u32, 70075, 22476, 220, 50, 59, 82, 49436, 79];
    for (i, &tok) in prefix.iter().enumerate() {
        let pred = model.forward_argmax(&mut gpu, tok);
        let ok = if pred == expected[i] { "✓" } else { "✗" };
        eprintln!("Token {}: {} → {} (expected {}) {}", i, tok, pred, expected[i], ok);
    }
}

/// Test: prefix + one encoder embed injection.
#[test]
fn verify_prefix_then_embed() {
    let _ = env_logger::try_init();
    let model_dir = std::path::Path::new("../../models/qwen3-asr-1.7b");
    if !model_dir.exists() { eprintln!("SKIP"); return; }

    let ref_path = "/tmp/enc_ref_17_2048.f32";
    if !std::path::Path::new(ref_path).exists() {
        eprintln!("SKIP: run test_split_encoder first");
        return;
    }

    let (mut gpu, mut model) = shady_thinker::asr_decoder::load_bf16_model(model_dir, 512);
    let h = model.config.hidden_size as usize;

    // Prefix
    let prefix = [151644u32, 8948, 198, 151645, 198, 151644, 872, 198, 151669];
    for &tok in &prefix {
        model.forward_argmax(&mut gpu, tok);
    }
    eprintln!("After prefix: seq_len={}", model.seq_len);

    // Read first encoder output token
    let ref_bytes = std::fs::read(ref_path).unwrap();
    let ref_data: &[f32] = bytemuck::cast_slice(&ref_bytes);
    let first_embed = &ref_data[..h];
    eprintln!("Encoder embed[0][0:4]: {:?}", &first_embed[..4]);

    // Inject it
    let t = model.forward_embed_argmax(&mut gpu, first_embed);
    eprintln!("After embed injection: predicted={}, seq_len={}", t, model.seq_len);

    // Also check: what does the same embed produce when used as the FIRST token?
    model.seq_len = 0;
    model.generated_tokens.clear();
    let t0 = model.forward_embed_argmax(&mut gpu, first_embed);
    eprintln!("Same embed as first token: predicted={}", t0);
}

/// Full pipeline with real encoder output, printing each generated token.
#[test]
fn verify_full_decode_real() {
    let _ = env_logger::try_init();
    let model_dir = std::path::Path::new("../../models/qwen3-asr-1.7b");
    if !model_dir.exists() { eprintln!("SKIP"); return; }
    let ref_path = "/tmp/enc_ref_17_2048.f32";
    if !std::path::Path::new(ref_path).exists() { eprintln!("SKIP"); return; }

    let (mut gpu, mut model) = shady_thinker::asr_decoder::load_bf16_model(model_dir, 512);
    let h = model.config.hidden_size as usize;

    let ref_bytes = std::fs::read(ref_path).unwrap();
    let enc_output: &[f32] = bytemuck::cast_slice(&ref_bytes);
    let enc_seq_len = 17u32;

    // Manually run the decode pipeline with logging
    model.seq_len = 0;
    model.generated_tokens.clear();

    // Prefix
    for &tok in &[151644u32, 8948, 198, 151645, 198, 151644, 872, 198, 151669] {
        model.forward_argmax(&mut gpu, tok);
    }
    eprintln!("After prefix: seq_len={}", model.seq_len);

    // Encoder output
    for i in 0..enc_seq_len as usize {
        let embed = &enc_output[i * h..(i + 1) * h];
        model.forward_embed_argmax(&mut gpu, embed);
    }
    eprintln!("After encoder: seq_len={}", model.seq_len);

    // Suffix (all but last)
    for &tok in &[151670u32, 151645, 198, 151644, 77091, 198] {
        model.forward_argmax(&mut gpu, tok);
    }
    eprintln!("After suffix (minus last): seq_len={}", model.seq_len);

    // Last token starts generation
    let mut token = model.forward_argmax(&mut gpu, 151704); // <|asr_text|>
    eprintln!("First generated: {} (seq_len={})", token, model.seq_len);

    let mut tokens = vec![];
    for i in 0..30 {
        if token == 151643 || token == 151645 { break; }
        tokens.push(token);
        token = model.forward_argmax(&mut gpu, token);
        eprintln!("  gen[{}]: {}", i, token);
    }
    eprintln!("Generated tokens: {:?}", tokens);
    // C decoder produces: "Hello. This is a test."
    // Expected tokens for this: ~[9707, 13, 1096, 374, 264, 1273, 13]
    // ("Hello" "." " This" " is" " a" " test" ".")
}

/// GPU encoder + decoder with longer audio (fox sentence).
#[test]
fn gpu_pipeline_fox() {
    let _ = env_logger::try_init();
    let model_dir = std::path::Path::new("../../models/qwen3-asr-1.7b");
    if !model_dir.exists() { eprintln!("SKIP"); return; }
    let stem_path = "/tmp/conv_stem_46_1024.f32";
    if !std::path::Path::new(stem_path).exists() { eprintln!("SKIP: run test_split_encoder first"); return; }

    let stem_bytes = std::fs::read(stem_path).unwrap();
    let stem: &[f32] = bytemuck::cast_slice(&stem_bytes);
    let seq_len = 46u32;

    // GPU encoder
    let mut encoder = shady_thinker::asr_encoder::AsrEncoder::new(model_dir);
    let enc_output = encoder.forward(stem, seq_len);
    eprintln!("GPU encoder: {} tokens", enc_output.len() / encoder.config.output_dim as usize);

    // GPU decoder
    let (mut gpu, mut model) = shady_thinker::asr_decoder::load_bf16_model(model_dir, 512);
    let tokens = shady_thinker::asr_decoder::gpu_asr_decode_tokens(&mut gpu, &mut model, &enc_output, seq_len);
    eprintln!("GPU decoder: {} tokens: {:?}", tokens.len(), &tokens);
}
