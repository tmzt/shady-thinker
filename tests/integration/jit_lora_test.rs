//! Integration tests for Qwen3.5 inference and JIT-LoRA learning.
//!
//! These require a GPU and model weights. Run with:
//!   MODEL_DIR=/path/to/qwen3.5 make test-integration

use std::path::PathBuf;

use shady_thinker::chat::*;

fn model_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("MODEL_DIR").expect("Set MODEL_DIR to a Qwen3.5 model directory"),
    )
}

// ── Inference tests (no learning) ─────────────────────────────────

/// Verify the model produces a coherent, correct answer.
#[test]
fn test_inference_capital_of_france() {
    let mut session = ChatSession::new(model_dir(), 128, false, None);
    let response = session.generate("What is the capital of France?", None);
    eprintln!("[france] {} tok: {:?}", response.token_count, response.text);
    let lower = response.text.to_lowercase();
    assert!(
        lower.contains("paris"),
        "Expected 'paris' in response, got: {}",
        response.text
    );
}

/// Verify the model can answer a different factual question.
#[test]
fn test_inference_largest_planet() {
    let mut session = ChatSession::new(model_dir(), 128, false, None);
    let response = session.generate("What is the largest planet in the solar system?", None);
    eprintln!("[planet] {} tok: {:?}", response.token_count, response.text);
    let lower = response.text.to_lowercase();
    assert!(
        lower.contains("jupiter"),
        "Expected 'jupiter' in response, got: {}",
        response.text
    );
}

/// Verify the model doesn't know a made-up term (baseline for learning tests).
#[test]
fn test_inference_unknown_term() {
    let mut session = ChatSession::new(model_dir(), 128, false, None);
    let response = session.generate("What is Blobly?", None);
    eprintln!("[blobly-baseline] {} tok: {:?}", response.token_count, response.text);
    let lower = response.text.to_lowercase();
    // Should NOT contain the specific fact we'd teach later
    let has_waterpark = lower.contains("waterpark") && lower.contains("wisconsin");
    assert!(
        !has_waterpark,
        "Model unexpectedly knew Blobly is a waterpark in Wisconsin: {}",
        response.text
    );
}

// ── Learning tests (require LoRA) ─────────────────────────────────

#[ignore]
#[test]
fn test_teach_and_recall() {
    let mut session = ChatSession::new(model_dir(), 128, true, None);

    let result = session
        .teach(
            "Q: What is Zorbex? A: Zorbex is a purple mineral from Mars",
            5,
        )
        .expect("teach should succeed");
    assert!(result.loss > 0.0);
    assert!(result.loss < 10.0);

    let response = session.generate("What is Zorbex?", None);
    let lower = response.text.to_lowercase();
    assert!(
        lower.contains("purple") || lower.contains("mineral") || lower.contains("mars"),
        "Expected recall of taught fact, got: {}",
        response.text
    );
}

#[ignore]
#[test]
fn test_teach_does_not_forget_anchors() {
    let mut session = ChatSession::new(model_dir(), 128, true, None);

    session.teach(
        "Q: What is Glorpium? A: A shiny crystal found in caves",
        5,
    );

    let response = session.generate("What is the capital of Wisconsin?", None);
    let lower = response.text.to_lowercase();
    assert!(
        lower.contains("madison"),
        "Expected anchor recall of Madison, got: {}",
        response.text
    );
}

#[ignore]
#[test]
fn test_forget_resets() {
    let mut session = ChatSession::new(model_dir(), 128, true, None);

    session.teach(
        "Q: What is Blipnox? A: A rare gemstone from Neptune",
        5,
    );
    session.forget();

    let response = session.generate("What is Blipnox?", None);
    let lower = response.text.to_lowercase();
    assert!(
        !lower.contains("neptune") || !lower.contains("gemstone"),
        "Expected forget to clear taught fact, but it was recalled: {}",
        response.text
    );
}

#[ignore]
#[test]
fn test_save_and_load_lora() {
    let dir = model_dir();
    let tmp_path = std::env::temp_dir().join("test_lora_checkpoint.safetensors");

    {
        let mut session = ChatSession::new(dir.clone(), 128, true, None);
        session.teach("Q: What is Quazzle? A: A fizzy drink from Saturn", 5);
        session.save_lora(&tmp_path);
    }

    {
        let mut session = ChatSession::new(dir, 128, false, Some(tmp_path.clone()));
        let response = session.generate("What is Quazzle?", None);
        let lower = response.text.to_lowercase();
        assert!(
            lower.contains("fizzy")
                || lower.contains("drink")
                || lower.contains("saturn"),
            "Expected recall from loaded LoRA, got: {}",
            response.text
        );
    }

    let _ = std::fs::remove_file(&tmp_path);
}

#[ignore]
#[test]
fn test_extract_learn_recall_blobly() {
    let mut session = ChatSession::new_with_extractor(
        model_dir(),
        128,
        None,
        Box::new(RegexFactExtractor),
    );

    let before = session.generate("What is Blobly?", None);
    let before_lower = before.text.to_lowercase();
    let knew_before =
        before_lower.contains("waterpark") && before_lower.contains("wisconsin");
    eprintln!(
        "[blobly pre-train] knew_before={}, response: {:?}",
        knew_before, before.text
    );

    let learn = session
        .learn_from_message("Q: What is Blobly?\nA: Blobly is a waterpark in Wisconsin");
    assert!(learn.trained);
    assert_eq!(learn.pairs.len(), 1);

    let after = session.generate("What is Blobly?", None);
    let after_lower = after.text.to_lowercase();
    assert!(
        after_lower.contains("waterpark") || after_lower.contains("wisconsin"),
        "Expected recall of 'waterpark in Wisconsin' after learning, got: {}",
        after.text
    );
}

#[ignore]
#[test]
fn test_learn_with_regex_extractor() {
    let mut session = ChatSession::new_with_extractor(
        model_dir(),
        128,
        None,
        Box::new(RegexFactExtractor),
    );

    let result =
        session.learn_from_message("Q: What is Flumzite?\nA: A glowing rock from Jupiter");
    assert!(result.trained);
    assert_eq!(result.pairs.len(), 1);

    let response = session.generate("What is Flumzite?", None);
    let lower = response.text.to_lowercase();
    assert!(
        lower.contains("glow") || lower.contains("rock") || lower.contains("jupiter"),
        "Expected recall of learned fact, got: {}",
        response.text
    );
}
