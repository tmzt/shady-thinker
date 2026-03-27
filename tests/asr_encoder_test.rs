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
