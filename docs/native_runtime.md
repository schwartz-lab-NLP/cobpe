# Native CoBPE runtime

The runtime is implemented in `rust_ext/compositional_runtime`. It is a separate
Rust crate with a PyO3 Python interface, packaged using maturin. It has no
Cargo dependency on `rustbpe` and does not invoke the BPE trainer.

The workflow is:

1. Python trains a BPE vocabulary using the external `rustbpe` package.
2. The metadata exporter builds the CoBPE composition inventory.
3. Python supplies the vocabulary, BPE pattern, and composition metadata to
   the native `CompositionalProcessor`.
4. Rust uses `tiktoken-rs` to encode text and our composition logic to produce
   base-token IDs and modifier rows. Reconstruction also runs in Rust.

The model receives these two streams. The extension does not require PyTorch
internally; Python connects it to the language-model backend.

## Setup

From the repository root, with `uv` and a stable Rust toolchain installed:

```bash
uv sync --locked --extra cpu
uv run --locked --extra cpu python -m cobpe check-install
```

Use `--extra gpu` in both commands for the NVIDIA GPU environment. `uv sync`
builds the local native extension automatically. No separate `maturin develop`
command is needed. Keep the selected extra consistent when invoking `uv run`.
The first installation needs network access to fetch Python packages and Rust
crates. The installation check then needs no datasets or model downloads.

## Source layout

- `src/lib.rs` exposes the Python module.
- `src/compositional/mod.rs` defines runtime configuration and processor state.
- `runtime.rs` builds the BPE runtime and metadata tables.
- `matching.rs` and `process.rs` implement compositional matching and processing.
- `surface.rs` reconstructs text from base tokens and modifier rows.
- `python.rs` exposes encoding, decoding, and batched operations to Python.

The import name `nanochat_compositional_rust` is retained for compatibility;
the source is part of this repository and built locally. `Cargo.lock` pins the
native dependency resolution.

## Development checks

```bash
uv sync --locked --extra cpu --group dev
uv run --locked --extra cpu python -m pytest tests/test_compositional.py tests/test_release_workflow.py
cargo test --locked --manifest-path rust_ext/compositional_runtime/Cargo.toml
```

The Python tests exercise the compiled native runtime. The Rust crate currently
has no Rust unit tests, so `cargo test` alone is a compilation check.
