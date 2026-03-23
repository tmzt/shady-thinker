//! Interactive chat with JIT LoRA learning — thin CLI wrapper.
//!
//! Usage:
//!   cargo run --release --features jit-lora --example chat -- <model_dir> [--max-tokens 256] [--learn] [--lora <path>]

use std::io::{self, Read as _, Write};
use std::os::unix::io::AsRawFd;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use tensorbend_rs::chat::ChatSession;

/// Set stdin to raw non-blocking mode, returns the old termios to restore later.
fn set_raw_nonblocking() -> libc::termios {
    unsafe {
        let fd = io::stdin().as_raw_fd();
        let mut old: libc::termios = std::mem::zeroed();
        libc::tcgetattr(fd, &mut old);
        let mut raw = old;
        raw.c_lflag &= !(libc::ICANON | libc::ECHO);
        raw.c_cc[libc::VMIN] = 0;
        raw.c_cc[libc::VTIME] = 0;
        libc::tcsetattr(fd, libc::TCSANOW, &raw);
        let flags = libc::fcntl(fd, libc::F_GETFL);
        libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK);
        old
    }
}

/// Restore terminal to previous mode.
fn restore_terminal(old: &libc::termios) {
    unsafe {
        let fd = io::stdin().as_raw_fd();
        libc::tcsetattr(fd, libc::TCSANOW, old);
        let flags = libc::fcntl(fd, libc::F_GETFL);
        libc::fcntl(fd, libc::F_SETFL, flags & !libc::O_NONBLOCK);
    }
}

/// Check if ESC was pressed (non-blocking).
fn esc_pressed(cancel: &AtomicBool) {
    let mut buf = [0u8; 8];
    match io::stdin().read(&mut buf) {
        Ok(n) if n > 0 => {
            if buf[..n].contains(&0x1B) {
                cancel.store(true, Ordering::Relaxed);
            }
        }
        _ => {}
    }
}

fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: {} <model_dir> [--max-tokens N] [--learn] [--lora <path>]",
            args[0]
        );
        eprintln!();
        eprintln!("  model_dir    Directory with safetensors, config.json, tokenizer.json");
        eprintln!("  --max-tokens Maximum response tokens (default: 256)");
        eprintln!("  --learn      Enable JIT LoRA learning from conversation");
        eprintln!("  --lora PATH  Load LoRA adapter from safetensors file");
        std::process::exit(1);
    }

    let model_dir = PathBuf::from(&args[1]);
    let mut max_tokens: u32 = 256;
    let mut learning_enabled = false;
    let mut lora_path: Option<PathBuf> = None;
    {
        let mut i = 2;
        while i < args.len() {
            match args[i].as_str() {
                "--max-tokens" if i + 1 < args.len() => {
                    max_tokens = args[i + 1].parse().unwrap_or(256);
                    i += 2;
                }
                "--learn" => {
                    learning_enabled = true;
                    i += 1;
                }
                "--lora" if i + 1 < args.len() => {
                    lora_path = Some(PathBuf::from(&args[i + 1]));
                    i += 2;
                }
                _ => i += 1,
            }
        }
    }

    eprintln!("Loading model from {:?}...", model_dir);
    let load_start = Instant::now();
    let mut session = ChatSession::new(model_dir, max_tokens, learning_enabled, lora_path);
    let load_elapsed = load_start.elapsed();
    eprintln!("Model loaded in {:.2}s", load_elapsed.as_secs_f64());

    if learning_enabled {
        eprintln!("Learning mode: ON — facts extracted automatically from conversation");
    }
    eprintln!();
    eprintln!("Commands:");
    eprintln!("  /teach <fact>              Learn a plain fact");
    eprintln!("  /teach Q: <question> A: <answer>  Learn a Q&A pair");
    eprintln!("  /forget                    Reset LoRA weights");
    eprintln!("  /save [path]               Checkpoint adapter to safetensors");
    eprintln!("  /quit                      Exit");
    eprintln!();

    loop {
        eprint!("\x1b[1mYou:\x1b[0m ");
        io::stderr().flush().unwrap();

        let mut user_input = String::new();
        if io::stdin().read_line(&mut user_input).unwrap() == 0 {
            break;
        }
        let user_input = user_input.trim().to_string();
        if user_input.is_empty() {
            continue;
        }

        // Commands
        if user_input == "/quit" || user_input == "/exit" {
            break;
        }
        if user_input == "/forget" {
            session.forget();
            eprintln!("\x1b[90m[LoRA reset — forgetting all learned facts]\x1b[0m");
            continue;
        }
        if user_input.starts_with("/save") {
            let save_path = user_input.strip_prefix("/save").unwrap().trim();
            let save_path = if save_path.is_empty() {
                "lora_adapter.safetensors"
            } else {
                save_path
            };
            session.save_lora(std::path::Path::new(save_path));
            eprintln!("\x1b[90m[Saved LoRA adapter to {:?}]\x1b[0m", save_path);
            continue;
        }
        if user_input.starts_with("/teach ") {
            let body = user_input.strip_prefix("/teach ").unwrap().trim();
            if let Some(result) = session.teach(body, 5) {
                eprintln!(
                    "\x1b[90m[Learned: Q=\"{}\" A=\"{}\" loss={:.3} step={}]\x1b[0m",
                    result.question, result.answer, result.loss, result.step
                );
            } else {
                eprintln!("Learning not enabled. Use --learn flag.");
            }
            continue;
        }

        // Auto-extract and learn
        let learn_start = Instant::now();
        let learn = session.learn_from_message(&user_input);
        if learn.trained {
            eprintln!(
                "\x1b[90m[Learned {} fact(s) in {:.1}s]\x1b[0m",
                learn.pairs.len(),
                learn_start.elapsed().as_secs_f64()
            );
            for (q, a) in &learn.pairs {
                eprintln!("\x1b[90m  Q: {} → A: {}\x1b[0m", q, a);
            }
        }

        // Generate response
        let cancel = AtomicBool::new(false);
        let old_term = set_raw_nonblocking();

        // Spawn a quick ESC check before generating
        // (In a real app you'd poll in a thread, but for simplicity we check inline)
        print!("\x1b[1mAI:\x1b[0m ");
        io::stdout().flush().unwrap();

        let result = session.generate(&user_input, Some(&cancel));
        restore_terminal(&old_term);

        if result.interrupted {
            eprintln!("\x1b[90m[interrupted]\x1b[0m");
        } else {
            println!("{}", result.text);
            eprintln!(
                "\x1b[90m[{} tok, {:.1} tok/s]\x1b[0m",
                result.token_count, result.tokens_per_sec
            );
        }
        eprintln!();
    }

    eprintln!("Bye!");
}
