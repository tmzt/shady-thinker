//! Build script for shady-thinker.
//!
//! If FAST_THINKER_MODEL_DIR is set and contains a `prefix_cache_*.bin` file,
//! the cache is embedded into the binary via `include_bytes!` so it travels
//! with the APK without requiring a separate file push to the device.
//!
//! Usage (macOS build):
//!   1. Run `shady-thinker cache-prefix <model_dir>` to generate the cache.
//!   2. Set FAST_THINKER_MODEL_DIR=<model_dir> when building the workspace:
//!        FAST_THINKER_MODEL_DIR=/path/to/model cargo build --release -p thinker_engine --features gpu

use std::path::{Path, PathBuf};

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let embed_rs = out_dir.join("prefix_cache_embed.rs");

    // Rerun if the model dir changes or a new cache appears.
    println!("cargo:rerun-if-env-changed=FAST_THINKER_MODEL_DIR");

    if let Ok(model_dir) = std::env::var("FAST_THINKER_MODEL_DIR") {
        if let Some(cache_path) = find_prefix_cache(Path::new(&model_dir)) {
            println!("cargo:rerun-if-changed={}", cache_path.display());
            let src = format!(
                "/// Prefix KV cache embedded at compile time from {:?}.\n\
                 pub static EMBEDDED_PREFIX_CACHE: Option<&[u8]> = \
                     Some(include_bytes!(\"{}\"));\n",
                cache_path,
                cache_path.display(),
            );
            std::fs::write(&embed_rs, src).expect("write prefix_cache_embed.rs");
            return;
        }
    }

    // No cache available — emit None so the code compiles regardless.
    std::fs::write(
        &embed_rs,
        "pub static EMBEDDED_PREFIX_CACHE: Option<&[u8]> = None;\n",
    )
    .expect("write prefix_cache_embed.rs");
}

fn find_prefix_cache(model_dir: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(model_dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("prefix_cache_") && name.ends_with(".bin") {
            return Some(entry.path());
        }
    }
    None
}
