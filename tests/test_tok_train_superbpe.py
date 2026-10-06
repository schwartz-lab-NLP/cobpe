import pytest
import importlib.util
from pathlib import Path
import sys
import json
import tiktoken
from tokenizers import Regex, pre_tokenizers
from transformers.convert_slow_tokenizer import bytes_to_unicode

TOK_TRAIN_PATH = Path(__file__).resolve().parents[1] / "scripts" / "tok_train.py"
ROOT = TOK_TRAIN_PATH.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("tok_train", TOK_TRAIN_PATH)
tok_train = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(tok_train)


def _make_args(**overrides):
    args = tok_train.build_parser().parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_superbpe_requires_explicit_transition_merges():
    args = _make_args(
        tokenizer_algorithm=tok_train.TOKENIZER_ALGORITHM_SUPERBPE,
        normalization_variant=tok_train.BASELINE_VARIANT,
        superbpe_transition_merges=None,
    )
    train_vocab_size = int(args.vocab_size + args.vocab_buffer_size)
    with pytest.raises(ValueError, match="--superbpe-transition-merges is required"):
        tok_train.validate_tokenizer_algorithm_args(args, train_vocab_size=train_vocab_size)


def test_superbpe_is_baseline_only():
    args = _make_args(
        tokenizer_algorithm=tok_train.TOKENIZER_ALGORITHM_SUPERBPE,
        normalization_variant=tok_train.SPACE_CAP_VARIANT,
        superbpe_transition_merges=100,
    )
    train_vocab_size = int(args.vocab_size + args.vocab_buffer_size)
    with pytest.raises(ValueError, match="baseline"):
        tok_train.validate_tokenizer_algorithm_args(args, train_vocab_size=train_vocab_size)


def test_superbpe_transition_merge_budget_validation():
    args = _make_args(
        tokenizer_algorithm=tok_train.TOKENIZER_ALGORITHM_SUPERBPE,
        normalization_variant=tok_train.BASELINE_VARIANT,
    )
    train_vocab_size = int(args.vocab_size + args.vocab_buffer_size)
    max_merges = tok_train.max_merge_operations_for_train_vocab(train_vocab_size)
    args.superbpe_transition_merges = int(max_merges + 1)
    with pytest.raises(ValueError, match="exceeds max merge budget"):
        tok_train.validate_tokenizer_algorithm_args(args, train_vocab_size=train_vocab_size)


def test_superbpe_requires_existing_external_trainer_script():
    args = _make_args(
        tokenizer_algorithm=tok_train.TOKENIZER_ALGORITHM_SUPERBPE,
        normalization_variant=tok_train.BASELINE_VARIANT,
        superbpe_transition_merges=10,
        superbpe_trainer_script="/tmp/definitely_missing_superbpe_trainer.py",
    )
    train_vocab_size = int(args.vocab_size + args.vocab_buffer_size)
    with pytest.raises(FileNotFoundError, match="superbpe-trainer-script"):
        tok_train.validate_tokenizer_algorithm_args(args, train_vocab_size=train_vocab_size)


def test_vocab_json_to_rustbpe_roundtrip(tmp_path):
    pattern = tok_train.SPLIT_PATTERN
    byte_encoder = bytes_to_unicode()
    vocab = {byte_encoder[i]: i for i in range(256)}
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")

    rust_tok = tok_train._vocab_json_to_rustbpe(str(vocab_path), pattern=pattern)
    assert rust_tok.get_vocab_size() == 256 + len(tok_train.SPECIAL_TOKENS)

    for text in [
        "Hello world!",
        "This is a tokenizer conversion parity check.",
        "Numbers and unicode: 42 🌍",
        "I'm testing contractions.\nnewline",
    ]:
        ids = rust_tok.encode(text)
        assert rust_tok.decode(ids) == text


def test_buffer_finalization_keeps_exact_target_and_uses_survivor_replacements():
    mergeable = {bytes([i]): i for i in range(256)}
    for offset in range(15):
        mergeable[bytes([ord("a"), ord("a") + offset])] = 256 + offset
    special_tokens = {
        token: len(mergeable) + idx
        for idx, token in enumerate(tok_train.SPECIAL_TOKENS)
    }
    enc = tiktoken.Encoding(
        name="buffer_finalization_probe",
        pat_str=tok_train.SPLIT_PATTERN,
        mergeable_ranks=mergeable,
        special_tokens=special_tokens,
    )
    tokenizer = tok_train.RustBPETokenizer(enc, "<|bos|>")
    assert tokenizer.get_vocab_size() == 280

    survivors = set(range(280)) - {256, 257}
    finalized, stats = tok_train.finalize_tokenizer_with_survivor_ids(
        tokenizer,
        survivor_ids=survivors,
        target_vocab_size=270,
        train_vocab_size=280,
    )

    assert finalized.get_vocab_size() == 270
    kept_token_bytes = set(finalized.enc._mergeable_ranks)
    assert bytes([ord("a"), ord("c")]) in kept_token_bytes  # old rank 258
    assert bytes([ord("a"), ord("a")]) not in kept_token_bytes  # removed survivor
    assert stats["selected_survivor_mergeables"] == 261
    assert stats["unreachable_survivors_skipped"] == 0


def test_buffer_finalization_skips_survivors_with_removed_merge_ancestors():
    mergeable = {bytes([i]): i for i in range(256)}
    mergeable[b"ab"] = 256
    mergeable[b"abc"] = 257
    mergeable[b"bc"] = 258
    mergeable[b"cd"] = 259
    mergeable[b"abcd"] = 260
    special_tokens = {
        token: len(mergeable) + idx
        for idx, token in enumerate(tok_train.SPECIAL_TOKENS)
    }
    enc = tiktoken.Encoding(
        name="buffer_finalization_ancestry_probe",
        pat_str=tok_train.SPLIT_PATTERN,
        mergeable_ranks=mergeable,
        special_tokens=special_tokens,
    )
    tokenizer = tok_train.RustBPETokenizer(enc, "<|bos|>")

    survivors = set(range(tokenizer.get_vocab_size())) - {256}
    finalized, stats = tok_train.finalize_tokenizer_with_survivor_ids(
        tokenizer,
        survivor_ids=survivors,
        target_vocab_size=267,
        train_vocab_size=270,
    )

    kept_token_bytes = set(finalized.enc._mergeable_ranks)
    assert b"abc" not in kept_token_bytes
    assert b"bc" in kept_token_bytes
    assert stats["unreachable_survivors_skipped"] == 1


def test_buffer_finalization_never_fills_from_decomposed_tokens():
    mergeable = {bytes([i]): i for i in range(256)}
    mergeable[b"ab"] = 256
    special_tokens = {
        token: len(mergeable) + idx
        for idx, token in enumerate(tok_train.SPECIAL_TOKENS)
    }
    enc = tiktoken.Encoding(
        name="buffer_finalization_insufficient_probe",
        pat_str=tok_train.SPLIT_PATTERN,
        mergeable_ranks=mergeable,
        special_tokens=special_tokens,
    )
    tokenizer = tok_train.RustBPETokenizer(enc, "<|bos|>")

    with pytest.raises(RuntimeError, match="does not contain enough reachable CoBPE survivors"):
        tok_train.finalize_tokenizer_with_survivor_ids(
            tokenizer,
            survivor_ids=set(range(256)) | set(range(257, 266)),
            target_vocab_size=266,
            train_vocab_size=266,
        )


def test_default_superbpe_stage2_pattern_allows_cross_space_merges():
    mergeable = {bytes([i]): i for i in range(256)}
    mergeable[b" a"] = 256
    enc = tiktoken.Encoding(
        name="superbpe_pattern_probe",
        pat_str=tok_train.DEFAULT_SUPERBPE_STAGE2_PATTERN,
        mergeable_ranks=mergeable,
        special_tokens={},
    )
    ids = enc.encode_ordinary(" a")
    assert len(ids) == 1
    assert enc.decode(ids) == " a"


def _split_with_pattern(pattern: str, text: str) -> list[str]:
    splitter = pre_tokenizers.Split(Regex(pattern), behavior="isolated", invert=False)
    return [piece for piece, _offsets in splitter.pre_tokenize_str(text)]


def test_baseline_split_keeps_contractions_and_hyphen_suffix_like_chunks():
    pattern = tok_train.SPLIT_PATTERN
    assert _split_with_pattern(pattern, "I'm") == ["I'm"]
    assert _split_with_pattern(pattern, "don't") == ["don't"]
    assert _split_with_pattern(pattern, "don’t") == ["don’t"]
    assert _split_with_pattern(pattern, "don`t") == ["don`t"]
    assert _split_with_pattern(pattern, "-less") == ["-less"]


def test_baseline_split_detaches_non_hyphen_boundary_punctuation():
    pattern = tok_train.SPLIT_PATTERN
    assert _split_with_pattern(pattern, "(word)") == ["(", "word", ")"]
    assert _split_with_pattern(pattern, "word,") == ["word", ","]
    assert _split_with_pattern(pattern, ".word") == [".", "word"]


def test_space_cap_split_keeps_contractions_but_detaches_hyphen():
    pattern = tok_train.WHITESPACE_ISOLATING_SPLIT_PATTERN
    assert _split_with_pattern(pattern, "I'm") == ["I'm"]
    assert _split_with_pattern(pattern, "don't") == ["don't"]
    assert _split_with_pattern(pattern, "don’t") == ["don’t"]
    assert _split_with_pattern(pattern, "don`t") == ["don`t"]
    assert _split_with_pattern(pattern, "-less") == ["-", "less"]
