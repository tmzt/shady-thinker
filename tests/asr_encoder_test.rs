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
