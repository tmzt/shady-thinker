use std::path::Path;

#[test]
fn load_and_forward_0_6b() {
    env_logger::init();

    let model_dir = Path::new("../../models/qwen3-asr-0.6b");
    if !model_dir.exists() {
        eprintln!("SKIP: model dir {:?} not found", model_dir);
        return;
    }

    eprintln!("Loading ASR encoder from {:?}", model_dir);
    let mut encoder = shady_thinker::asr_encoder::AsrEncoder::new(model_dir);
    eprintln!("Config: {:?}", encoder.config);

    // Synthetic input: 10 tokens of d_model dimensions (as if from conv stem)
    let seq_len = 10u32;
    let d_model = encoder.config.d_model;
    let input: Vec<f32> = (0..seq_len * d_model).map(|i| (i as f32 * 0.001).sin()).collect();

    eprintln!("Running forward pass: seq_len={}, d_model={}", seq_len, d_model);
    let output = encoder.forward(&input, seq_len);

    let out_dim = encoder.config.output_dim;
    assert_eq!(output.len(), (seq_len * out_dim) as usize,
        "output size mismatch: expected {}x{}={}", seq_len, out_dim, seq_len * out_dim);

    // Check output is not all zeros (sanity)
    let sum: f32 = output.iter().map(|x| x.abs()).sum();
    eprintln!("Output sum of abs: {:.4} (should be nonzero)", sum);
    assert!(sum > 0.0, "output is all zeros");

    eprintln!("PASS: {} output values, sum={:.4}", output.len(), sum);
}
