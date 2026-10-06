"""
Train a RustBPE tokenizer on nanochat parquet data.

Supports baseline tokenization and space/case normalization for CoBPE.
"""

import argparse
from collections import deque
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch
import tiktoken

from nanochat.common import get_base_dir
from nanochat.dataset import parquets_iter_batched
from nanochat.tokenizer import (
    RustBPETokenizer,
    SPECIAL_TOKENS,
    SPLIT_PATTERN,
    WHITESPACE_ISOLATING_SPLIT_PATTERN,
)


BASELINE_VARIANT = "baseline"
SPACE_CAP_VARIANT = "space_cap"
NORMALIZATION_VARIANTS = [BASELINE_VARIANT, SPACE_CAP_VARIANT]
CASE_NORMALIZATION_CHOICES = ["none", "first_letter", "all_letters", "all_letters_except_all_caps"]
TOKENIZER_ALGORITHM_BPE = "bpe"
TOKENIZER_ALGORITHM_SUPERBPE = "superbpe"
TOKENIZER_ALGORITHM_CHOICES = [TOKENIZER_ALGORITHM_BPE, TOKENIZER_ALGORITHM_SUPERBPE]

# Stage-2 separator regex from upstream SuperBPE repo (superword training mode).
# Upstream uses HF Split with "isolated", where non-matching spans are preserved
# as chunks. tiktoken instead tokenizes only regex matches, so we append an
# explicit complementary branch to keep non-separator spans and preserve full
# coverage while still enabling cross-space merges.
SUPERBPE_STAGE2_SEPARATOR_PATTERN = r"""\p{N}{1,3}| ?[^\s\p{L}\p{N}]{2,}[\r\n/]*| +(?!\S)"""
DEFAULT_SUPERBPE_STAGE2_PATTERN = (
    rf"""{SUPERBPE_STAGE2_SEPARATOR_PATTERN}|(?:[\s\S]+?(?=(?:{SUPERBPE_STAGE2_SEPARATOR_PATTERN}|\z)))"""
)

WORD_PATTERN = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")

TextTransform = Callable[[str], str]
def _vocab_json_to_rustbpe(vocab_json_path: str, pattern: str) -> RustBPETokenizer:
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    with open(vocab_json_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError(f"Unexpected/empty vocab JSON: {vocab_json_path}")

    byte_encoder = bytes_to_unicode()
    byte_decoder = {encoded: raw for raw, encoded in byte_encoder.items()}

    mergeable_ranks: Dict[bytes, int] = {}
    for token_surface, rank in vocab.items():
        if not isinstance(token_surface, str):
            raise ValueError(f"Unexpected non-string token surface in vocab: {type(token_surface)}")
        try:
            token_bytes = bytes(byte_decoder[ch] for ch in token_surface)
        except KeyError as exc:
            raise ValueError(
                f"Token surface contains characters outside ByteLevel alphabet: {token_surface!r}"
            ) from exc
        mergeable_ranks[token_bytes] = int(rank)

    if len(mergeable_ranks) != len(vocab):
        raise ValueError(
            f"Decoded mergeable ranks size mismatch: decoded={len(mergeable_ranks)} vocab={len(vocab)}"
        )

    special_tokens = {
        name: len(mergeable_ranks) + i
        for i, name in enumerate(SPECIAL_TOKENS)
    }
    enc = tiktoken.Encoding(
        name="superbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )
    return RustBPETokenizer(enc, "<|bos|>")


def _resolve_superbpe_python_bin(value: Optional[str]) -> str:
    explicit = (value or "").strip()
    if explicit:
        return explicit
    from_env = os.environ.get("SUPERBPE_PYTHON_BIN", "").strip()
    if from_env:
        return from_env
    return sys.executable


def _default_superbpe_external_script() -> str:
    return os.path.join(os.path.dirname(__file__), "superbpe_external_trainer.py")


def _write_superbpe_training_jsonl(
    *,
    path: str,
    text_iterator_factory,
    progress_every_docs: int = 25_000,
    progress_every_chars: int = 50_000_000,
) -> Dict[str, object]:
    ndocs = 0
    nchars = 0
    last_report_chars = 0
    start_time = time.time()
    digest = hashlib.sha256()
    with open(path, "w", encoding="utf-8") as f:
        for text in text_iterator_factory():
            if not isinstance(text, str):
                raise TypeError(f"Expected string text, got {type(text)}")
            row = {"text": text}
            line = json.dumps(row, ensure_ascii=False)
            f.write(line)
            f.write("\n")
            ndocs += 1
            nchars += len(text)
            digest.update(text.encode("utf-8"))
            digest.update(b"\n")
            should_report_docs = progress_every_docs > 0 and ndocs % progress_every_docs == 0
            should_report_chars = progress_every_chars > 0 and (nchars - last_report_chars) >= progress_every_chars
            if should_report_docs or should_report_chars:
                elapsed = max(1e-9, time.time() - start_time)
                chars_per_sec = nchars / elapsed
                print(
                    "[SuperBPE prep] "
                    f"docs={ndocs:,} chars={nchars:,} elapsed={elapsed:.1f}s "
                    f"rate={chars_per_sec:,.0f} chars/s",
                    flush=True,
                )
                last_report_chars = nchars
    elapsed = time.time() - start_time
    return {
        "docs": int(ndocs),
        "chars": int(nchars),
        "elapsed_seconds": float(elapsed),
        "sha256": digest.hexdigest(),
    }


def _run_superbpe_external_trainer(
    *,
    python_bin: str,
    script_path: str,
    input_jsonl: str,
    output_dir: str,
    vocab_size_no_special: int,
    transition_merges: int,
    stage1_pattern: str,
    stage2_pattern: str,
) -> Dict[str, object]:
    cmd = [
        python_bin,
        script_path,
        "--input-jsonl", input_jsonl,
        "--output-dir", output_dir,
        "--vocab-size-no-special", str(vocab_size_no_special),
        "--transition-merges", str(transition_merges),
        "--stage1-pattern", stage1_pattern,
        "--stage2-pattern", stage2_pattern,
    ]
    print("[SuperBPE train] launching external trainer:", flush=True)
    print("  " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    trainer_log_path = os.path.join(output_dir, "superbpe_external_stdout.log")
    output_tail = deque(maxlen=80)
    progress_re = re.compile(r"\[\s*\d+\s*/\s*\d+\s*\]")
    logged_lines = 0
    start_time = time.time()
    last_heartbeat = start_time
    last_progress_line = ""
    assert proc.stdout is not None
    with open(trainer_log_path, "w", encoding="utf-8") as log_f:
        for raw_line in proc.stdout:
            log_f.write(raw_line)
            line = raw_line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            output_tail.append(line)
            logged_lines += 1
            is_high_signal = (
                line.startswith("Step ")
                or line.startswith("Calling ")
                or line.startswith("In do_train_extend")
                or line.startswith("Length of merges")
                or "fork probe" in line.lower()
                or bool(progress_re.search(line))
            )
            if is_high_signal:
                print(f"[superbpe-trainer] {line}", flush=True)
                if progress_re.search(line):
                    last_progress_line = line

            now = time.time()
            if now - last_heartbeat >= 60.0:
                elapsed = now - start_time
                heartbeat = (
                    f"[superbpe-trainer] heartbeat: elapsed={elapsed:.0f}s "
                    f"lines={logged_lines}"
                )
                if last_progress_line:
                    heartbeat += f" last_progress='{last_progress_line}'"
                print(heartbeat, flush=True)
                last_heartbeat = now
    returncode = proc.wait()
    elapsed = time.time() - start_time
    print(f"[SuperBPE train] external trainer finished in {elapsed:.1f}s", flush=True)
    print(f"[SuperBPE train] external trainer log: {trainer_log_path}", flush=True)
    if returncode != 0:
        stdout_tail = "\n".join(list(output_tail)[-40:])
        env_hint = ""
        if "ModuleNotFoundError" in stdout_tail or "No module named 'tokenizers'" in stdout_tail:
            env_hint = (
                "\nThe external environment appears missing required modules. "
                "Ensure it has tokenizers-superbpe installed."
            )
        raise RuntimeError(
            "External SuperBPE trainer failed.\n"
            f"return_code={returncode}\n"
            f"command={' '.join(cmd)}\n"
            f"log_path={trainer_log_path}\n"
            f"stdout_tail:\n{stdout_tail}\n"
            + env_hint
        )

    manifest_path = os.path.join(output_dir, "superbpe_external_manifest.json")
    if not os.path.exists(manifest_path):
        raise RuntimeError(
            f"External SuperBPE trainer did not produce manifest: {manifest_path}"
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not bool(manifest.get("fork_probe_ok", False)):
        raise RuntimeError(
            "External SuperBPE trainer reported fork probe failure. "
            "This usually means stock tokenizers behavior (no merges.txt inheritance) "
            "or mismatched external environment."
        )
    manifest["external_command"] = cmd
    manifest["trainer_stdout_log"] = trainer_log_path
    return manifest


def train_superbpe_tokenizer(
    *,
    text_iterator_factory,
    vocab_size: int,
    stage1_pattern: str,
    stage2_pattern: str,
    transition_merges: int,
    python_bin: str,
    trainer_script: str,
    keep_temp_data: bool,
) -> Tuple[RustBPETokenizer, Dict[str, object]]:

    if transition_merges < 0:
        raise ValueError(f"transition_merges must be >= 0, got {transition_merges}")

    # We follow RustBPETokenizer semantics: specials are appended after training.
    vocab_size_no_special = int(vocab_size - len(SPECIAL_TOKENS))
    if vocab_size_no_special < 256:
        raise ValueError(
            f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        )

    stage1_vocab_size = 256 + int(transition_merges)
    if stage1_vocab_size > vocab_size_no_special:
        raise ValueError(
            f"Stage-1 vocab size ({stage1_vocab_size}) exceeds final mergeable size ({vocab_size_no_special})."
        )

    if not os.path.exists(trainer_script):
        raise FileNotFoundError(
            f"SuperBPE external trainer script not found: {trainer_script}"
        )

    if keep_temp_data:
        temp_dir_obj = tempfile.mkdtemp(prefix="tok_train_superbpe_")
        temp_dir_ctx = None
        temp_dir = temp_dir_obj
    else:
        temp_dir_ctx = tempfile.TemporaryDirectory(prefix="tok_train_superbpe_")
        temp_dir = temp_dir_ctx.__enter__()

    try:
        input_jsonl = os.path.join(temp_dir, "superbpe_train.jsonl")
        output_dir = os.path.join(temp_dir, "superbpe_out")
        os.makedirs(output_dir, exist_ok=True)

        print("[SuperBPE train] stage 0/2: materializing training JSONL...", flush=True)
        prep_stats = _write_superbpe_training_jsonl(
            path=input_jsonl,
            text_iterator_factory=text_iterator_factory,
        )
        print(
            "[SuperBPE train] stage 0/2 complete: "
            f"docs={int(prep_stats['docs']):,} chars={int(prep_stats['chars']):,} "
            f"elapsed={float(prep_stats.get('elapsed_seconds', 0.0)):.1f}s",
            flush=True,
        )
        print(f"[SuperBPE train] jsonl path: {input_jsonl}", flush=True)
        if int(prep_stats["docs"]) <= 0:
            raise RuntimeError(
                "No training documents were produced after preprocessing; cannot train SuperBPE."
            )

        print("[SuperBPE train] stages 1/2 + 2/2: external SuperBPE trainer...", flush=True)
        external_manifest = _run_superbpe_external_trainer(
            python_bin=python_bin,
            script_path=trainer_script,
            input_jsonl=input_jsonl,
            output_dir=output_dir,
            vocab_size_no_special=int(vocab_size_no_special),
            transition_merges=int(transition_merges),
            stage1_pattern=str(stage1_pattern),
            stage2_pattern=str(stage2_pattern),
        )
        vocab_json_path = os.path.join(output_dir, "final-vocab.json")
        if not os.path.exists(vocab_json_path):
            raise RuntimeError(
                f"External SuperBPE output missing final vocab file: {vocab_json_path}"
            )
        tokenizer = _vocab_json_to_rustbpe(vocab_json_path, pattern=stage2_pattern)
        manifest = {
            "version": 1,
            "python_bin": python_bin,
            "trainer_script": os.path.realpath(trainer_script),
            "input_jsonl": os.path.realpath(input_jsonl),
            "output_dir": os.path.realpath(output_dir),
            "prep_stats": prep_stats,
            "external_manifest": external_manifest,
        }
        return tokenizer, manifest
    finally:
        if temp_dir_ctx is not None:
            temp_dir_ctx.__exit__(None, None, None)


def lowercase_first_letter_per_word(text: str) -> str:
    def _replace(match: re.Match) -> str:
        word = match.group(0)
        return word[0].lower() + word[1:]

    return WORD_PATTERN.sub(_replace, text)


def lowercase_all_letters_per_word(text: str) -> str:
    def _replace(match: re.Match) -> str:
        return match.group(0).lower()

    return WORD_PATTERN.sub(_replace, text)


def lowercase_all_letters_except_all_caps_words(text: str) -> str:
    def _replace(match: re.Match) -> str:
        word = match.group(0)
        if word.isupper():
            return word
        if word[0].isupper() and word[1:].islower():
            return word.lower()
        return word

    return WORD_PATTERN.sub(_replace, text)


def build_text_preprocessor(args) -> TextTransform:
    transforms: List[TextTransform] = []

    if args.normalization_variant == SPACE_CAP_VARIANT:
        if args.case_normalization == "first_letter":
            transforms.append(lowercase_first_letter_per_word)
        elif args.case_normalization == "all_letters":
            transforms.append(lowercase_all_letters_per_word)
        elif args.case_normalization == "all_letters_except_all_caps":
            transforms.append(lowercase_all_letters_except_all_caps_words)
        else:
            raise ValueError(f"Unsupported case normalization mode: {args.case_normalization}")

    if not transforms:
        return lambda text: text

    def _apply(text: str) -> str:
        output = text
        for transform in transforms:
            output = transform(output)
        return output

    return _apply


def resolve_case_normalization(normalization_variant: str, case_normalization: str) -> str:
    if normalization_variant == BASELINE_VARIANT:
        return "none"
    if case_normalization == "none":
        raise ValueError(
            "--case-normalization=none is only valid with --normalization-variant baseline"
        )
    return case_normalization


def resolve_training_split_pattern(normalization_variant: str) -> Tuple[Optional[str], str]:
    if normalization_variant == SPACE_CAP_VARIANT:
        return WHITESPACE_ISOLATING_SPLIT_PATTERN, "whitespace_isolating"
    return None, "gpt4_default"


def max_merge_operations_for_train_vocab(train_vocab_size: int) -> int:
    vocab_size_no_special = int(train_vocab_size - len(SPECIAL_TOKENS))
    if vocab_size_no_special < 256:
        return -1
    return int(vocab_size_no_special - 256)


def validate_tokenizer_algorithm_args(args, train_vocab_size: int) -> None:
    algorithm = str(args.tokenizer_algorithm)
    if algorithm not in TOKENIZER_ALGORITHM_CHOICES:
        raise ValueError(
            f"Unsupported tokenizer algorithm: {algorithm}. "
            f"Supported: {TOKENIZER_ALGORITHM_CHOICES}"
        )

    if algorithm == TOKENIZER_ALGORITHM_BPE:
        if args.superbpe_transition_merges is not None:
            raise ValueError(
                "--superbpe-transition-merges is only valid with --tokenizer-algorithm superbpe."
            )
        return

    # SuperBPE validations.
    if args.normalization_variant != BASELINE_VARIANT:
        raise ValueError(
            "SuperBPE baseline is currently limited to --normalization-variant baseline."
        )
    if args.superbpe_transition_merges is None:
        raise ValueError(
            "--superbpe-transition-merges is required with --tokenizer-algorithm superbpe."
        )
    t = int(args.superbpe_transition_merges)
    if t < 0:
        raise ValueError("--superbpe-transition-merges must be >= 0.")
    max_merges = max_merge_operations_for_train_vocab(train_vocab_size)
    if max_merges < 0:
        raise ValueError(
            f"Train vocab size {train_vocab_size} is too small for byte-level BPE with specials."
        )
    if t > max_merges:
        raise ValueError(
            f"--superbpe-transition-merges={t} exceeds max merge budget {max_merges} "
            f"for train_vocab_size={train_vocab_size}."
        )
    script_path = os.path.abspath(os.path.expanduser(str(args.superbpe_trainer_script)))
    if not os.path.exists(script_path):
        raise FileNotFoundError(
            f"--superbpe-trainer-script does not exist: {script_path}"
        )


def _export_rust_tokenizer_to_hf_dir(rust_tokenizer: RustBPETokenizer, output_dir: str) -> None:
    from transformers import PreTrainedTokenizerFast
    from scripts.export_hf_tokenizer_from_rustbpe import _build_backend_tokenizer

    enc = rust_tokenizer.enc
    mergeable_ranks = getattr(enc, "_mergeable_ranks", None)
    pattern = getattr(enc, "_pat_str", None)
    special_tokens_map = getattr(enc, "_special_tokens", None)
    if not isinstance(mergeable_ranks, dict) or not pattern or not isinstance(special_tokens_map, dict):
        raise ValueError("Unexpected RustBPETokenizer encoding internals for HF export.")

    special_tokens_in_id_order = [t for t, _ in sorted(special_tokens_map.items(), key=lambda kv: kv[1])]
    backend = _build_backend_tokenizer(
        mergeable_ranks=mergeable_ranks,
        pattern=pattern,
        special_tokens_in_id_order=special_tokens_in_id_order,
    )
    os.makedirs(output_dir, exist_ok=True)
    tokenizer_json = os.path.join(output_dir, "tokenizer.json")
    backend.save(tokenizer_json)

    ordered_specials = [t for t in SPECIAL_TOKENS if t in special_tokens_in_id_order]
    extras = [t for t in special_tokens_in_id_order if t not in ordered_specials]
    additional_special_tokens = ordered_specials + extras
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_json,
        bos_token="<|bos|>",
        eos_token="<|bos|>",
        additional_special_tokens=additional_special_tokens,
    )
    hf_tok.save_pretrained(output_dir)


def finalize_tokenizer_with_survivor_ids(
    tokenizer: RustBPETokenizer,
    *,
    survivor_ids,
    target_vocab_size: int,
    train_vocab_size: int,
) -> Tuple[RustBPETokenizer, Dict[str, int]]:
    """Compact a buffered RustBPE tokenizer to exactly the selected target size."""
    enc = tokenizer.enc
    mergeable_ranks = getattr(enc, "_mergeable_ranks", None)
    pattern = getattr(enc, "_pat_str", None)
    special_tokens_map = getattr(enc, "_special_tokens", None)
    if not isinstance(mergeable_ranks, dict) or not pattern or not isinstance(special_tokens_map, dict):
        raise ValueError("Unexpected tokenizer encoding internals while finalizing buffer tokens.")

    num_special = int(len(special_tokens_map))
    if target_vocab_size <= num_special:
        raise ValueError(
            f"target vocab size ({target_vocab_size}) must be larger than number of special tokens ({num_special})."
        )
    target_mergeable = int(target_vocab_size - num_special)
    existing_mergeable = int(len(mergeable_ranks))
    if target_mergeable > existing_mergeable:
        raise ValueError(
            f"Target mergeable size ({target_mergeable}) exceeds trained mergeable size ({existing_mergeable})."
        )
    if target_mergeable < 256:
        raise ValueError(
            f"Target mergeable size ({target_mergeable}) is below byte alphabet size 256."
        )

    rank_to_token = {int(rank): token_bytes for token_bytes, rank in mergeable_ranks.items()}
    survivor_rank_set = {
        int(r) for r in survivor_ids if 0 <= int(r) < existing_mergeable
    }
    survivor_mergeable_ranks = sorted(r for r in survivor_rank_set if r >= 256)
    kept_ranks = set(range(256))
    kept_token_bytes = {rank_to_token[rank] for rank in kept_ranks}
    unreachable_survivors = 0

    for rank in survivor_mergeable_ranks:
        if len(kept_ranks) >= target_mergeable:
            break
        token_bytes = rank_to_token[rank]
        # A retained BPE token must still have a construction after earlier
        # decomposition candidates are removed. Otherwise it becomes a dead
        # model row that the compact tokenizer can never emit.
        reachable = any(
            token_bytes[:split] in kept_token_bytes
            and token_bytes[split:] in kept_token_bytes
            for split in range(1, len(token_bytes))
        )
        if not reachable:
            unreachable_survivors += 1
            continue
        kept_ranks.add(rank)
        kept_token_bytes.add(token_bytes)

    if len(kept_ranks) != target_mergeable:
        raise RuntimeError(
            "The tokenizer buffer does not contain enough reachable CoBPE survivors: "
            f"selected={len(kept_ranks)} expected={target_mergeable} "
            f"unreachable_survivors={unreachable_survivors}. Increase --vocab-buffer-size."
        )

    selected_old_ranks = sorted(kept_ranks)
    new_mergeable_ranks = {
        rank_to_token[old_rank]: new_rank
        for new_rank, old_rank in enumerate(selected_old_ranks)
    }

    special_tokens_in_id_order = [t for t, _ in sorted(special_tokens_map.items(), key=lambda kv: kv[1])]
    new_special_tokens = {
        token: len(new_mergeable_ranks) + i
        for i, token in enumerate(special_tokens_in_id_order)
    }

    finalized_enc = tiktoken.Encoding(
        name=getattr(enc, "name", "rustbpe"),
        pat_str=pattern,
        mergeable_ranks=new_mergeable_ranks,
        special_tokens=new_special_tokens,
    )
    finalized = RustBPETokenizer(finalized_enc, "<|bos|>")

    stats = {
        "target_vocab_size": int(target_vocab_size),
        "train_vocab_size": int(train_vocab_size),
        "num_special_tokens": int(num_special),
        "train_mergeable_size": int(existing_mergeable),
        "target_mergeable_size": int(target_mergeable),
        "selected_survivor_mergeables": int(sum(1 for r in selected_old_ranks if r in survivor_rank_set)),
        "unreachable_survivors_skipped": int(unreachable_survivors),
        "survivor_full_vocab": int(train_vocab_size),
        "survivor_base_vocab": int(len(set(int(v) for v in survivor_ids))),
        "survivor_removed": int(train_vocab_size - len(set(int(v) for v in survivor_ids))),
    }
    return finalized, stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
    parser.add_argument(
        "--output-base-dir",
        type=str,
        default="",
        help="Root directory for tokenizer outputs (overrides NANOCHAT_BASE_DIR).",
    )
    parser.add_argument(
        "--local-parquet-dir",
        type=str,
        default="",
        help=(
            "Explicit directory containing local parquet shards for tokenizer training. "
            "If empty, falls back to $LOCAL_PARQUET_DIR."
        ),
    )
    parser.add_argument("--max-chars", type=int, default=2_000_000_000, help="Maximum characters to train on (default: 2B)")
    parser.add_argument("--doc-cap", type=int, default=10_000, help="Maximum characters per document (default: 10,000)")
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32768,
        help="Target vocabulary size (default: 32768 = 2^15). "
             "When --vocab-buffer-size > 0, training uses vocab-size + buffer.",
    )
    parser.add_argument(
        "--vocab-buffer-size",
        type=int,
        default=0,
        help="Extra tokenizer slots to train above --vocab-size "
             "(useful when downstream CoBPE decomposition removes some tokens).",
    )
    parser.add_argument("--tokenizer-dirname", type=str, default="tokenizer",
                        help="Output directory name under NANOCHAT_BASE_DIR (default: tokenizer)")
    parser.add_argument("--normalization-variant", type=str, default=BASELINE_VARIANT, choices=NORMALIZATION_VARIANTS,
                        help="Text normalization mode for tokenizer training")
    parser.add_argument("--case-normalization", type=str, default="all_letters_except_all_caps",
                        choices=CASE_NORMALIZATION_CHOICES,
                        help="Case normalization mode (baseline always resolves to 'none')")
    parser.add_argument("--seed", type=int, default=42, help="Torch seed (for reproducibility bookkeeping)")
    parser.add_argument(
        "--tokenizer-algorithm",
        type=str,
        default=TOKENIZER_ALGORITHM_BPE,
        choices=TOKENIZER_ALGORITHM_CHOICES,
        help="Tokenizer algorithm to train (default: bpe)",
    )
    parser.add_argument(
        "--superbpe-transition-merges",
        type=int,
        default=None,
        help=(
            "SuperBPE transition point t in number of inherited stage-1 merge operations. "
            "Required when --tokenizer-algorithm=superbpe."
        ),
    )
    parser.add_argument(
        "--superbpe-stage2-pattern",
        type=str,
        default=DEFAULT_SUPERBPE_STAGE2_PATTERN,
        help=(
            "Stage-2 pretokenization regex used by SuperBPE after transition t "
            "(default: upstream SuperBPE pattern)."
        ),
    )
    parser.add_argument(
        "--superbpe-python-bin",
        type=str,
        default="",
        help=(
            "Python executable for external SuperBPE trainer process. "
            "Default: $SUPERBPE_PYTHON_BIN, else current interpreter."
        ),
    )
    parser.add_argument(
        "--superbpe-trainer-script",
        type=str,
        default=_default_superbpe_external_script(),
        help="Path to external SuperBPE trainer script.",
    )
    parser.add_argument(
        "--superbpe-keep-temp-data",
        action="store_true",
        help="Keep temporary SuperBPE JSONL/artifacts for debugging.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.output_base_dir.strip():
        os.environ["NANOCHAT_BASE_DIR"] = os.path.abspath(os.path.expanduser(args.output_base_dir))
    local_parquet_dir = args.local_parquet_dir.strip()
    if not local_parquet_dir:
        local_parquet_dir = os.environ.get("LOCAL_PARQUET_DIR", "").strip()
    if local_parquet_dir:
        local_parquet_dir = os.path.abspath(os.path.expanduser(local_parquet_dir))
        if not os.path.isdir(local_parquet_dir):
            raise FileNotFoundError(
                f"--local-parquet-dir does not exist (or is not a directory): {local_parquet_dir}"
            )
    args.superbpe_python_bin = _resolve_superbpe_python_bin(args.superbpe_python_bin)
    args.superbpe_trainer_script = os.path.abspath(os.path.expanduser(args.superbpe_trainer_script))
    if args.vocab_buffer_size < 0:
        raise ValueError("--vocab-buffer-size must be >= 0")
    train_vocab_size = int(args.vocab_size + args.vocab_buffer_size)
    if train_vocab_size <= 0:
        raise ValueError("effective train vocab size must be > 0")
    args.case_normalization = resolve_case_normalization(
        normalization_variant=args.normalization_variant,
        case_normalization=args.case_normalization,
    )
    validate_tokenizer_algorithm_args(args, train_vocab_size=train_vocab_size)

    print(f"max_chars: {args.max_chars:,}")
    print(f"doc_cap: {args.doc_cap:,}")
    print(f"output_base_dir: {get_base_dir()}")
    print(
        "local_parquet_dir: "
        f"{local_parquet_dir if local_parquet_dir else '(dataset default resolution)'}"
    )
    print(f"vocab_size_target: {args.vocab_size:,}")
    print(f"vocab_buffer_size: {args.vocab_buffer_size:,}")
    print(f"vocab_size_train: {train_vocab_size:,}")
    print(f"tokenizer_dirname: {args.tokenizer_dirname}")
    print(f"normalization_variant: {args.normalization_variant}")
    print(f"case_normalization: {args.case_normalization}")
    print(f"tokenizer_algorithm: {args.tokenizer_algorithm}")
    if args.tokenizer_algorithm == TOKENIZER_ALGORITHM_SUPERBPE:
        print(f"superbpe_transition_merges: {args.superbpe_transition_merges}")
        print(f"superbpe_stage2_pattern: {args.superbpe_stage2_pattern}")
        print(f"superbpe_python_bin: {args.superbpe_python_bin}")
        print(f"superbpe_trainer_script: {args.superbpe_trainer_script}")
        print(f"superbpe_keep_temp_data: {bool(args.superbpe_keep_temp_data)}")
    print(f"seed: {args.seed}")

    torch.manual_seed(args.seed)

    preprocess_fn = build_text_preprocessor(args)
    split_pattern, split_pattern_mode = resolve_training_split_pattern(args.normalization_variant)
    if args.tokenizer_algorithm == TOKENIZER_ALGORITHM_SUPERBPE:
        stage1_mode = split_pattern_mode if split_pattern is not None else "gpt4_default"
        print(f"split_pattern_mode_stage1: {stage1_mode}")
        print("split_pattern_mode_stage2: superbpe")
    else:
        print(f"split_pattern_mode: {split_pattern_mode}")

    # -------------------------------------------------------------------------
    # Text iterator
    def text_iterator():
        """
        1) Flatten the batches into a single iterator
        2) Crop every document to args.doc_cap characters
        3) Optionally apply normalization transform
        4) Break when we've seen args.max_chars raw characters
        """
        nchars = 0
        for batch in parquets_iter_batched(split="train", data_dir=local_parquet_dir or None):
            for doc in batch:
                doc_text = doc
                if len(doc_text) > args.doc_cap:
                    doc_text = doc_text[:args.doc_cap]
                nchars += len(doc_text)
                processed_text = preprocess_fn(doc_text)
                if processed_text:
                    yield processed_text
                if nchars > args.max_chars:
                    return

    # -------------------------------------------------------------------------
    # Train the tokenizer
    t0 = time.time()
    superbpe_manifest: Optional[Dict[str, object]] = None
    if args.tokenizer_algorithm == TOKENIZER_ALGORITHM_SUPERBPE:
        tokenizer, superbpe_manifest = train_superbpe_tokenizer(
            text_iterator_factory=text_iterator,
            vocab_size=int(train_vocab_size),
            stage1_pattern=split_pattern if split_pattern is not None else SPLIT_PATTERN,
            stage2_pattern=str(args.superbpe_stage2_pattern),
            transition_merges=int(args.superbpe_transition_merges),
            python_bin=str(args.superbpe_python_bin),
            trainer_script=str(args.superbpe_trainer_script),
            keep_temp_data=bool(args.superbpe_keep_temp_data),
        )
    else:
        tokenizer = RustBPETokenizer.train_from_iterator(
            text_iterator(),
            train_vocab_size,
            pattern=split_pattern,
        )
    t1 = time.time()
    train_time = t1 - t0
    print(f"Training time: {train_time:.2f}s")

    # Save the tokenizer to disk
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, args.tokenizer_dirname)
    tokenizer.save(tokenizer_dir)
    if superbpe_manifest is not None:
        superbpe_manifest_path = os.path.join(tokenizer_dir, "superbpe_manifest.json")
        with open(superbpe_manifest_path, "w", encoding="utf-8") as f:
            json.dump(superbpe_manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"Saved SuperBPE manifest to {superbpe_manifest_path}")

    # Persist buffer metadata so downstream metadata export can finalize exactly
    # with the intended decomposition settings.
    if int(args.vocab_buffer_size) > 0:
        buffer_meta = {
            "version": 1,
            "vocab_size_target": int(args.vocab_size),
            "vocab_buffer_size": int(args.vocab_buffer_size),
            "vocab_size_train": int(train_vocab_size),
            "normalization_variant": str(args.normalization_variant),
            "case_normalization": str(args.case_normalization),
            "tokenizer_algorithm": str(args.tokenizer_algorithm),
            "superbpe_transition_merges": (
                int(args.superbpe_transition_merges)
                if args.superbpe_transition_merges is not None
                else None
            ),
            "superbpe_stage2_pattern": (
                str(args.superbpe_stage2_pattern)
                if args.tokenizer_algorithm == TOKENIZER_ALGORITHM_SUPERBPE
                else None
            ),
            "superbpe_python_bin": (
                str(args.superbpe_python_bin)
                if args.tokenizer_algorithm == TOKENIZER_ALGORITHM_SUPERBPE
                else None
            ),
            "superbpe_trainer_script": (
                str(args.superbpe_trainer_script)
                if args.tokenizer_algorithm == TOKENIZER_ALGORITHM_SUPERBPE
                else None
            ),
            "seed": int(args.seed),
            # Finalization requires the exact requested CoBPE decomposition and
            # is therefore mandatory in export_compositional_metadata.
            "finalized_at_metadata_export": False,
        }
        buffer_meta_path = os.path.join(tokenizer_dir, "tokenizer_buffer.json")
        with open(buffer_meta_path, "w", encoding="utf-8") as f:
            json.dump(buffer_meta, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"Saved VD buffer config to {buffer_meta_path}")
    # -------------------------------------------------------------------------
    # Quick inline sanity check
    test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    assert decoded == test_text

    # -------------------------------------------------------------------------
    # Cache token_bytes for bits-per-byte evaluation
    vocab_size = tokenizer.get_vocab_size()
    special_set = set(tokenizer.get_special_tokens())
    token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
    token_bytes = []
    for token_id in range(vocab_size):
        token_str = token_strings[token_id]
        if token_str in special_set:
            token_bytes.append(0)
        else:
            token_bytes.append(len(token_str.encode("utf-8")))
    token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device="cpu")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    with open(token_bytes_path, "wb") as f:
        torch.save(token_bytes, f)
    print(f"Saved token_bytes to {token_bytes_path}")

    # -------------------------------------------------------------------------
    # Log to report
    from nanochat.report import get_report

    token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
    get_report().log(section="Tokenizer training", data=[
        vars(args),
        {"train_time": train_time},
        {"num_special_tokens": len(special_set)},
        {
            "token_bytes_min": int(token_bytes_nonzero.min().item()),
            "token_bytes_max": int(token_bytes_nonzero.max().item()),
            "token_bytes_mean": token_bytes_nonzero.mean().item(),
            "token_bytes_std": token_bytes_nonzero.std().item(),
        },
    ])


if __name__ == "__main__":
    main()
