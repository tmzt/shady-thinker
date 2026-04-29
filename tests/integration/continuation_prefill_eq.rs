//! Numerical equivalence: continuation prefill (`pos_offset > 0`) vs.
//! from-scratch prefill of the same combined token sequence must
//! produce the same final-position logits.
//!
//! This is the regression check for the `pos_offset` change in
//! `Model::prefill_gptq` and the qknorm/RoPE/causal-attn shaders. If
//! the math diverges (RoPE position misalignment, KV cache write
//! offset wrong, DeltaNet hist/state restore incomplete, attention
//! K/V buffer-source swap), this test catches it before the
//! gibberish snowball reaches the classifier.
//!
//! Skipped (returns Ok) when `MODEL_DIR` is unset or missing, so CI
//! without a model checked out passes silently. Run with
//! `MODEL_DIR=/path/to/model cargo test -p shady-thinker --test continuation_prefill_eq -- --nocapture`.

use std::path::PathBuf;

use shady_thinker::inference::InferenceSession;

fn model_dir() -> Option<PathBuf> {
    let p = std::env::var("MODEL_DIR").ok().map(PathBuf::from)?;
    if p.exists() { Some(p) } else { None }
}

fn read_logits(session: &mut InferenceSession) -> Vec<f32> {
    session.gpu.flush_and_wait();
    let vocab = session.model.config.vocab_size as u64;
    let bytes = session.gpu.read_buffer(&session.model.state.logits, vocab * 4);
    bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
}

fn linf(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

#[test]
fn continuation_matches_from_scratch_logits() {
    let dir = match model_dir() {
        Some(d) => d,
        None => { eprintln!("SKIP: MODEL_DIR not set or missing"); return; }
    };

    let _ = env_logger::builder().is_test(true).try_init();

    // 16 + 8 token split — small enough to fit any prefill batch
    // limit, large enough that pos_offset != 0 for every query token.
    let prefix_ids: Vec<u32> = (100u32..116).collect();
    let query_ids: Vec<u32> = (200u32..208).collect();
    let combined: Vec<u32> = prefix_ids.iter().chain(query_ids.iter()).copied().collect();

    let mut session = InferenceSession::new(dir, 256);

    // GPTQ-only path. The bf16 prefill takes a different code path
    // (legacy unbatched `prefill()`) and isn't what we're verifying.
    if session.model.bf16_mode {
        eprintln!("SKIP: model is bf16 — continuation prefill is gptq-only");
        return;
    }

    // ── Path A: from-scratch prefill of the full combined sequence ──
    session.model.seq_len = 0;
    session.model.generated_tokens.clear();
    session.model.prefill_gptq(&mut session.gpu, &combined, 0);
    session.gpu.flush_and_wait();
    session.model.dispatch_lm_head(&mut session.gpu);
    let logits_full = read_logits(&mut session);
    eprintln!("[eq] from-scratch logits: len={} max={:.4} min={:.4}",
        logits_full.len(),
        logits_full.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
        logits_full.iter().fold(f32::INFINITY, |a, &b| a.min(b)));

    // ── Path B: set_prefix(prefix), restore, then continuation prefill ──
    session.set_prefix(&prefix_ids);
    session.restore_prefix_snapshot();
    session.model.prefill_gptq(&mut session.gpu, &query_ids, session.prefix_len);
    session.gpu.flush_and_wait();
    session.model.dispatch_lm_head(&mut session.gpu);
    let logits_cont = read_logits(&mut session);
    eprintln!("[eq] continuation  logits: len={} max={:.4} min={:.4}",
        logits_cont.len(),
        logits_cont.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
        logits_cont.iter().fold(f32::INFINITY, |a, &b| a.min(b)));

    let diff = linf(&logits_full, &logits_cont);
    eprintln!("[eq] ||full - cont||_∞ = {:.6}", diff);

    // Tolerance: int4 + bf16 numerics + non-deterministic GPU
    // reductions add up; same architectural path should still land
    // within 1e-2 on the ∞-norm. If it drifts much past that, the
    // RoPE / KV / DeltaNet alignment is off.
    assert!(
        diff < 1e-2,
        "continuation prefill diverges from from-scratch: ∞-norm = {diff} (tolerance 1e-2)",
    );

    // Argmax sanity: at temp=0 the sampled token must agree.
    let argmax = |v: &[f32]| -> u32 {
        v.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32).unwrap()
    };
    let tok_full = argmax(&logits_full);
    let tok_cont = argmax(&logits_cont);
    assert_eq!(tok_full, tok_cont,
        "argmax token disagrees between paths: full={tok_full} cont={tok_cont}");
    eprintln!("[eq] argmax agrees: token={tok_full}");

    // ── Path C: a SECOND continuation prefill with the same shapes ──
    // Repro for the cross-request gibberish snowball: the
    // bind_group_cache key is `pipeline + buf_addr + buf_size`. Per-call
    // local prefill buffers (`pg_residual` …) drop at function end, and
    // wgpu can hand back the same Rust struct address with the same
    // size for a same-shaped query in the next call. Without dropping
    // the `pg_*` cache entries, the second call's dispatches read stale
    // GPU memory from the first call's now-reused buffers, producing
    // all-zero / gibberish output — but only when the buffer sizes
    // match (different chunk lengths dodge it via size-key mismatch,
    // which is why the original 3-file batch worked while a same-size
    // 4th call snowballed).
    session.restore_prefix_snapshot();
    session.model.prefill_gptq(&mut session.gpu, &query_ids, session.prefix_len);
    session.gpu.flush_and_wait();
    session.model.dispatch_lm_head(&mut session.gpu);
    let logits_cont2 = read_logits(&mut session);
    let diff2 = linf(&logits_full, &logits_cont2);
    eprintln!("[eq] ||full - cont2||_∞ = {:.6}", diff2);
    assert!(
        diff2 < 1e-2,
        "second continuation diverges from from-scratch: ∞-norm = {diff2} (tolerance 1e-2)",
    );
    let tok_cont2 = argmax(&logits_cont2);
    assert_eq!(tok_full, tok_cont2,
        "argmax token disagrees on second continuation: full={tok_full} cont2={tok_cont2}");
    eprintln!("[eq] second continuation argmax agrees: token={tok_full}");
}
