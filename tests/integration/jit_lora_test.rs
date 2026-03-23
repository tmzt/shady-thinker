//! Integration tests for JIT-LoRA learning and recall.
//!
//! These require a GPU and model weights. Set MODEL_DIR to run:
//!   MODEL_DIR=/path/to/qwen3.5 cargo test --features jit-lora --test jit_lora_test

use std::path::PathBuf;

use tensorbend_rs::chat::*;

fn model_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("MODEL_DIR").expect("Set MODEL_DIR to a Qwen3.5 model directory"),
    )
}

fn make_session() -> ChatSession {
    ChatSession::new(model_dir(), 64, true, None)
}

#[test]
fn test_teach_and_recall() {
    let mut session = make_session();

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

#[test]
fn test_teach_does_not_forget_anchors() {
    let mut session = make_session();

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

#[test]
fn test_forget_resets() {
    let mut session = make_session();

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

#[test]
fn test_save_and_load_lora() {
    let dir = model_dir();
    let tmp_path = std::env::temp_dir().join("test_lora_checkpoint.safetensors");

    {
        let mut session = ChatSession::new(dir.clone(), 64, true, None);
        session.teach("Q: What is Quazzle? A: A fizzy drink from Saturn", 5);
        session.save_lora(&tmp_path);
    }

    {
        let mut session = ChatSession::new(dir, 64, false, Some(tmp_path.clone()));
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

/// End-to-end: baseline → learn → recall.
/// Blobly appears in the extraction few-shot prompt but the model
/// should NOT know what it is before training.
#[test]
fn test_extract_learn_recall_blobly() {
    let mut session = ChatSession::new_with_extractor(
        model_dir(),
        64,
        None,
        Box::new(RegexFactExtractor),
    );

    // Before learning — model should not know
    let before = session.generate("What is Blobly?", None);
    let before_lower = before.text.to_lowercase();
    let knew_before =
        before_lower.contains("waterpark") && before_lower.contains("wisconsin");
    eprintln!(
        "[blobly pre-train] knew_before={}, response: {:?}",
        knew_before, before.text
    );

    // Learn
    let learn = session
        .learn_from_message("Q: What is Blobly?\nA: Blobly is a waterpark in Wisconsin");
    assert!(learn.trained);
    assert_eq!(learn.pairs.len(), 1);

    // After learning — model should recall
    let after = session.generate("What is Blobly?", None);
    let after_lower = after.text.to_lowercase();
    assert!(
        after_lower.contains("waterpark") || after_lower.contains("wisconsin"),
        "Expected recall of 'waterpark in Wisconsin' after learning, got: {}",
        after.text
    );
}

#[test]
fn test_learn_with_regex_extractor() {
    let mut session = ChatSession::new_with_extractor(
        model_dir(),
        64,
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
