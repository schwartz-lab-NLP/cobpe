"""
Optional Rust-backed compositional runtime bridge.

This module keeps the Python side very small:
- try to load a compiled Rust processor
- pass a compact JSON config derived from compositional metadata
- normalize the returned ids / modifier rows
"""

from __future__ import annotations

import json
import hashlib
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

try:
    import nanochat_compositional_rust as _nanochat_compositional_rust  # type: ignore[import-not-found]
except Exception:
    _nanochat_compositional_rust = None


def _normalize_result(result: Any) -> tuple[list[int], list[list[int]]]:
    if isinstance(result, dict):
        token_ids = [int(v) for v in result["output_ids"]]
        modifier_rows = [[int(x) for x in row] for row in result["modifier_rows"]]
        return token_ids, modifier_rows
    token_ids, modifier_rows = result
    return [int(v) for v in token_ids], [[int(x) for x in row] for row in modifier_rows]


class RustCompositionalBackend:
    def __init__(self, processor: Any):
        self._processor = processor

    def process_text(self, text: str) -> tuple[list[int], list[list[int]]]:
        return _normalize_result(self._processor.process_text(text))

    def process_text_batch(self, texts: list[str]) -> list[tuple[list[int], list[list[int]]]]:
        return [_normalize_result(item) for item in self._processor.process_text_batch(texts)]

    def decode_with_modifiers(self, token_ids: list[int], modifier_rows: list[list[int]]) -> str:
        return str(self._processor.decode_with_modifiers(token_ids, modifier_rows))

    def decode_token_with_modifiers(self, token_id: int, modifier_row: list[int]) -> str:
        return str(self._processor.decode_with_modifiers([int(token_id)], [[int(v) for v in modifier_row]]))

    def utf8_len_with_modifiers_batch(
        self,
        token_ids: list[int],
        modifier_rows: list[list[int]],
    ) -> list[int]:
        return [int(v) for v in self._processor.utf8_len_with_modifiers_batch(token_ids, modifier_rows)]

    def debug_tokenize_text(self, text: str) -> dict[str, Any]:
        return json.loads(self._processor.debug_tokenize_text_json(text))

    def debug_process_text(self, text: str) -> dict[str, Any]:
        return json.loads(self._processor.debug_process_text_json(text))

    def encode_text(self, text: str) -> list[int]:
        return [int(v) for v in self._processor.encode_text(text)]

    @property
    def build_info(self) -> dict[str, Any]:
        module = _nanochat_compositional_rust
        return {
            "extension": "nanochat_compositional_rust",
            "version": getattr(module, "__version__", None) if module is not None else None,
            "build": getattr(module, "__build__", None) if module is not None else None,
            "module_path": str(getattr(module, "__file__", "")) if module is not None else None,
        }


def rust_extension_info() -> dict[str, Any]:
    """Return stable, reportable information about the optional extension."""
    module = _nanochat_compositional_rust
    if module is None or not hasattr(module, "CompositionalProcessor"):
        return {
            "available": False,
            "extension": "nanochat_compositional_rust",
            "version": None,
            "build": None,
            "module_path": None,
        }
    return {
        "available": True,
        "extension": "nanochat_compositional_rust",
        "version": getattr(module, "__version__", None),
        "build": getattr(module, "__build__", None),
        "module_path": str(getattr(module, "__file__", "")),
    }


@lru_cache(maxsize=1)
def _byte_to_unicode() -> dict[int, str]:
    """Match the byte alphabet used by tokenizers' ByteLevel pre-tokenizer."""
    direct = list(range(ord("!"), ord("~") + 1))
    direct += list(range(ord("¡"), ord("¬") + 1))
    direct += list(range(ord("®"), ord("ÿ") + 1))
    remap = direct[:]
    extra = 0
    for value in range(256):
        if value not in direct:
            direct.append(value)
            remap.append(256 + extra)
            extra += 1
    return {byte: chr(codepoint) for byte, codepoint in zip(direct, remap)}


@lru_cache(maxsize=1)
def _bytelevel_unicode_to_byte() -> dict[str, int]:
    return {value: byte for byte, value in _byte_to_unicode().items()}


def _bytelevel_token_to_bytes(token: str) -> bytes:
    inverse = _bytelevel_unicode_to_byte()
    try:
        return bytes(inverse[char] for char in token)
    except KeyError as exc:
        raise ValueError(
            f"HF ByteLevel token contains a character outside the reversible byte alphabet: {token!r}"
        ) from exc


def tokenizer_json_to_base_bpe(tokenizer_json: str) -> dict[str, Any]:
    """Convert a local HF ByteLevel BPE without changing IDs or merge ranks.

    The Rust runtime consumes tiktoken's byte/rank representation.  HF's
    ByteLevel vocabulary is a reversible display alphabet, so conversion is
    lossless only for the exact BPE shape emitted by the analysis.
    """
    payload = json.loads(tokenizer_json)
    model = payload.get("model") or {}
    if model.get("type") != "BPE":
        raise ValueError("Rust analysis requires an HF BPE tokenizer.json")
    if model.get("byte_fallback") is not True:
        raise ValueError("Rust analysis requires byte_fallback=true for exact UTF-8 behavior")
    pre = payload.get("pre_tokenizer") or {}
    if pre.get("type") != "Sequence" or len(pre.get("pretokenizers", [])) != 2:
        raise ValueError("Rust analysis requires the analysis Split+ByteLevel pre-tokenizer")
    split, bytelevel = pre["pretokenizers"]
    pattern = (split.get("pattern") or {}).get("Regex")
    if not isinstance(pattern, str) or split.get("behavior") != "Isolated":
        raise ValueError("Rust analysis could not recover the HF split regex")
    if bytelevel.get("type") != "ByteLevel" or bytelevel.get("add_prefix_space"):
        raise ValueError("Rust analysis requires ByteLevel(add_prefix_space=false)")
    vocab = model.get("vocab") or {}
    if not vocab:
        raise ValueError("HF tokenizer.json has an empty BPE vocabulary")
    ids = sorted(int(value) for value in vocab.values())
    if ids != list(range(len(ids))) or len(set(vocab.values())) != len(vocab):
        raise ValueError("HF vocabulary IDs are not contiguous; exact Rust ID parity is unavailable")
    mergeable_ranks = []
    for token, token_id in sorted(vocab.items(), key=lambda item: int(item[1])):
        if not isinstance(token, str):
            raise ValueError("HF BPE vocabulary contains a non-string token")
        mergeable_ranks.append(
            {"token": _bytelevel_token_to_bytes(token).decode("latin-1"), "rank": int(token_id)}
        )
    token_bytes = {token: _bytelevel_token_to_bytes(token) for token in vocab}
    merges = model.get("merges") or []
    for merge_index, merge in enumerate(merges):
        if isinstance(merge, str):
            parts = merge.split(" ", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid HF BPE merge at index {merge_index}")
            left, right = parts
        else:
            if len(merge) != 2:
                raise ValueError(f"Invalid HF BPE merge at index {merge_index}")
            left, right = merge
        merged = left + right
        if left not in token_bytes or right not in token_bytes or merged not in token_bytes:
            raise ValueError(f"HF BPE merge at index {merge_index} is not representable losslessly")
        if int(vocab[merged]) < 256:
            raise ValueError("HF BPE merge target has a byte-alphabet ID; IDs/ranks cannot be preserved")
    return {
        "pattern": pattern,
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": {},
    }


def tokenizer_fingerprint(tokenizer_dir: Optional[str]) -> Optional[str]:
    if tokenizer_dir is None:
        return None
    path = Path(tokenizer_dir) / "tokenizer.json"
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _extract_base_bpe_config(tokenizer_dir: Optional[str]) -> Optional[dict[str, Any]]:
    if tokenizer_dir is None:
        return None
    pickle_path = Path(tokenizer_dir) / "tokenizer.pkl"
    if not pickle_path.exists():
        return None
    try:
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
    except Exception:
        return None

    mergeable_ranks = getattr(enc, "_mergeable_ranks", None)
    pattern = getattr(enc, "_pat_str", None)
    special_tokens = getattr(enc, "_special_tokens", None)
    if not isinstance(mergeable_ranks, dict) or not isinstance(pattern, str):
        return None

    rank_entries = []
    for token_bytes, rank in sorted(mergeable_ranks.items(), key=lambda kv: int(kv[1])):
        if not isinstance(token_bytes, (bytes, bytearray)):
            return None
        rank_entries.append(
            {
                "token": bytes(token_bytes).decode("latin-1"),
                "rank": int(rank),
            }
        )

    special_token_map: dict[str, int] = {}
    if isinstance(special_tokens, dict):
        for token, token_id in special_tokens.items():
            special_token_map[str(token)] = int(token_id)

    return {
        "pattern": pattern,
        "mergeable_ranks": rank_entries,
        "special_tokens": special_token_map,
    }


def build_rust_backend(
    spec,
    *,
    tokenizer_dir: Optional[str] = None,
    allow_hf_bpe_conversion: bool = False,
) -> Optional[RustCompositionalBackend]:
    if _nanochat_compositional_rust is None or not hasattr(_nanochat_compositional_rust, "CompositionalProcessor"):
        return None

    tokenizer_json = None
    base_bpe = None
    if tokenizer_dir is not None:
        tokenizer_json_path = Path(tokenizer_dir) / "tokenizer.json"
        if tokenizer_json_path.exists():
            tokenizer_json = tokenizer_json_path.read_text(encoding="utf-8")
            if allow_hf_bpe_conversion:
                base_bpe = tokenizer_json_to_base_bpe(tokenizer_json)

    payload = spec.to_rust_config(tokenizer_json=tokenizer_json)
    if base_bpe is None:
        base_bpe = _extract_base_bpe_config(tokenizer_dir)
    if base_bpe is not None:
        payload["base_bpe"] = base_bpe
    if not payload.get("token_meta") and not payload.get("tokenizer_json") and base_bpe is None:
        return None

    config_json = json.dumps(payload, separators=(",", ":"))
    processor = _nanochat_compositional_rust.CompositionalProcessor(config_json)
    return RustCompositionalBackend(processor)
