MODEL_DIR ?= ./models

.PHONY: test test-unit test-integration check build

# Run unit tests only (no GPU/model needed)
test-unit:
	cargo test --features jit-lora --lib

# Run integration tests (requires model weights in MODEL_DIR)
test-integration:
	MODEL_DIR=$(MODEL_DIR) cargo test --features jit-lora --test jit_lora_test -- --nocapture --test-threads=1

# Run all tests
test: test-unit test-integration

# Type-check everything
check:
	cargo check
	cargo check --features jit-lora
	cargo check --features jit-lora --example chat

# Build release
build:
	cargo build --release --features jit-lora
