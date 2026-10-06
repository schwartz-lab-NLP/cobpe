"""
Export nanochat RustBPETokenizer (tokenizer.pkl) to a Hugging Face fast tokenizer.

This is useful for metadata-building utilities that expect an HF tokenizer path.

From project root:
  python -m scripts.export_hf_tokenizer_from_rustbpe
"""

import argparse
import os
import pickle
import shutil
from dataclasses import dataclass
from typing import Iterable, List, Optional

from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.convert_slow_tokenizer import bytes_to_unicode
from tokenizers import AddedToken, Regex, Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import BPE

from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS


def _token_bytes_to_string(token_bytes: bytes, byte_encoder: dict[int, str]) -> str:
    # Mirrors transformers TikTokenConverter behavior.
    return "".join([byte_encoder[ord(ch)] for ch in token_bytes.decode("latin-1")])


def _extract_vocab_and_merges(mergeable_ranks: dict[bytes, int]) -> tuple[dict[str, int], list[tuple[str, str]]]:
    """
    Convert tiktoken mergeable ranks into tokenizers.BPE vocab+merges.
    This reproduces the logic used by HF TikTokenConverter but avoids blobfile.
    """
    byte_encoder = bytes_to_unicode()
    vocab: dict[str, int] = {}
    merges_local: list[tuple[bytes, bytes, int]] = []
    for token, rank in mergeable_ranks.items():
        vocab[_token_bytes_to_string(token, byte_encoder)] = rank
        if len(token) == 1:
            continue
        local: list[tuple[bytes, bytes, int]] = []
        for idx in range(1, len(token)):
            left = token[:idx]
            right = token[idx:]
            if left in mergeable_ranks and right in mergeable_ranks and (left + right) in mergeable_ranks:
                local.append((left, right, rank))
        local = sorted(local, key=lambda x: (mergeable_ranks[x[0]], mergeable_ranks[x[1]]))
        merges_local.extend(local)
    merges_local = sorted(merges_local, key=lambda x: x[2])
    merges = [
        (
            _token_bytes_to_string(left, byte_encoder),
            _token_bytes_to_string(right, byte_encoder),
        )
        for (left, right, _) in merges_local
    ]
    return vocab, merges


def _build_backend_tokenizer(
    *,
    mergeable_ranks: dict[bytes, int],
    pattern: str,
    special_tokens_in_id_order: list[str],
    byte_fallback: bool = False,
) -> Tokenizer:
    vocab, merges = _extract_vocab_and_merges(mergeable_ranks)
    tokenizer = Tokenizer(BPE(vocab, merges, fuse_unk=False, byte_fallback=byte_fallback))
    if hasattr(tokenizer.model, "ignore_merges"):
        tokenizer.model.ignore_merges = True
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(pattern), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(
        [AddedToken(tok, normalized=False, special=True) for tok in special_tokens_in_id_order]
    )
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    return tokenizer


def _default_check_texts() -> List[str]:
    return [
        "Hello world!",
        "Numbers: 12 345 6789",
        "Contractions: I'm, you're, it's.",
        "Unicode: 你好世界 🌍",
        "Punctuation: (a+b)=c; wow?!",
        "Whitespace\tand\nnewlines.",
    ]


def _iter_dataset_texts(limit: int) -> Iterable[str]:
    if limit <= 0:
        return []
    from nanochat.dataset import parquets_iter_batched

    seen = 0
    # Use val split for deterministic small sampling.
    for batch in parquets_iter_batched("val"):
        for doc in batch:
            yield doc
            seen += 1
            if seen >= limit:
                return


@dataclass
class ParityReport:
    checked: int
    id_mismatches: int
    decode_mismatches: int
    special_id_mismatches: int


def _check_parity(
    rust_tok: RustBPETokenizer,
    hf_tok: PreTrainedTokenizerFast,
    texts: List[str],
    special_tokens: List[str],
    strict: bool,
) -> ParityReport:
    id_mismatches = 0
    decode_mismatches = 0
    for text in texts:
        rust_ids = rust_tok.encode(text)
        hf_ids = hf_tok.encode(text, add_special_tokens=False)
        if rust_ids != hf_ids:
            id_mismatches += 1
            if strict:
                raise ValueError(
                    "Tokenizer id parity mismatch.\n"
                    f"text={text!r}\n"
                    f"rust={rust_ids[:32]}\n"
                    f"hf={hf_ids[:32]}"
                )

        rust_dec = rust_tok.decode(rust_ids)
        hf_dec = hf_tok.decode(hf_ids, skip_special_tokens=False)
        if rust_dec != hf_dec:
            decode_mismatches += 1
            if strict:
                raise ValueError(
                    "Tokenizer decode parity mismatch.\n"
                    f"text={text!r}\n"
                    f"rust_dec={rust_dec!r}\n"
                    f"hf_dec={hf_dec!r}"
                )

    special_id_mismatches = 0
    for tok in special_tokens:
        rust_id = rust_tok.encode_special(tok)
        hf_id = hf_tok.convert_tokens_to_ids(tok)
        if rust_id != hf_id:
            special_id_mismatches += 1
            if strict:
                raise ValueError(
                    f"Special token id mismatch for {tok!r}: rust={rust_id}, hf={hf_id}"
                )

    return ParityReport(
        checked=len(texts),
        id_mismatches=id_mismatches,
        decode_mismatches=decode_mismatches,
        special_id_mismatches=special_id_mismatches,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export nanochat RustBPETokenizer to HF fast tokenizer")
    parser.add_argument(
        "--source-tokenizer-dir",
        type=str,
        default="",
        help="Directory containing tokenizer.pkl (default: $NANOCHAT_BASE_DIR/tokenizer)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output HF tokenizer directory (default: $NANOCHAT_BASE_DIR/tokenizer_hf)",
    )
    parser.add_argument(
        "--bos-token",
        type=str,
        default="<|bos|>",
        help="HF bos_token to set",
    )
    parser.add_argument(
        "--eos-token",
        type=str,
        default="<|bos|>",
        help="HF eos_token to set (default mirrors doc-separator token used by nanochat)",
    )
    parser.add_argument(
        "--dataset-check-docs",
        type=int,
        default=0,
        help="Also run parity checks on first N validation documents (0 disables)",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Do not fail on parity mismatches; only report counts",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output-dir if it exists",
    )
    args = parser.parse_args()

    base_dir = get_base_dir()
    source_dir = args.source_tokenizer_dir or os.path.join(base_dir, "tokenizer")
    output_dir = args.output_dir or os.path.join(base_dir, "tokenizer_hf")
    source_pkl = os.path.join(source_dir, "tokenizer.pkl")
    if not os.path.exists(source_pkl):
        raise FileNotFoundError(f"tokenizer.pkl not found at: {source_pkl}")

    if os.path.exists(output_dir):
        if not args.overwrite:
            raise FileExistsError(
                f"Output dir already exists: {output_dir}. Use --overwrite to replace."
            )
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    with open(source_pkl, "rb") as f:
        enc = pickle.load(f)

    mergeable_ranks = getattr(enc, "_mergeable_ranks", None)
    pattern = getattr(enc, "_pat_str", None)
    special_tokens_map = getattr(enc, "_special_tokens", None)
    if not isinstance(mergeable_ranks, dict) or not pattern or not isinstance(special_tokens_map, dict):
        raise ValueError("Unexpected encoding object in tokenizer.pkl; missing tiktoken internals.")

    # 1) Convert tiktoken encoding to tokenizers.Tokenizer, then wrap as HF fast tokenizer
    special_tokens_in_id_order = [t for t, _ in sorted(special_tokens_map.items(), key=lambda kv: kv[1])]
    backend = _build_backend_tokenizer(
        mergeable_ranks=mergeable_ranks,
        pattern=pattern,
        special_tokens_in_id_order=special_tokens_in_id_order,
    )
    tokenizer_json = os.path.join(output_dir, "tokenizer.json")
    backend.save(tokenizer_json)

    # Keep order stable: known nanochat specials first, then any extras.
    ordered_specials = [t for t in SPECIAL_TOKENS if t in special_tokens_in_id_order]
    extras = [t for t in special_tokens_in_id_order if t not in ordered_specials]
    additional_special_tokens = ordered_specials + extras

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_json,
        bos_token=args.bos_token if args.bos_token else None,
        eos_token=args.eos_token if args.eos_token else None,
        additional_special_tokens=additional_special_tokens,
    )
    hf_tok.save_pretrained(output_dir)

    # Propagate source tokenizer artifacts needed by downstream tooling.
    source_tokenizer_pkl = os.path.join(source_dir, "tokenizer.pkl")
    if os.path.exists(source_tokenizer_pkl):
        shutil.copy2(source_tokenizer_pkl, os.path.join(output_dir, "tokenizer.pkl"))
    source_buffer_meta = os.path.join(source_dir, "tokenizer_buffer.json")
    if not os.path.exists(source_buffer_meta):
        source_buffer_meta = os.path.join(source_dir, "vd_buffer_config.json")
    if os.path.exists(source_buffer_meta):
        shutil.copy2(source_buffer_meta, os.path.join(output_dir, "tokenizer_buffer.json"))

    # 2) Parity checks against source RustBPETokenizer
    rust_tok = RustBPETokenizer.from_directory(source_dir)
    hf_loaded = AutoTokenizer.from_pretrained(output_dir, use_fast=True)
    check_texts = _default_check_texts()
    if args.dataset_check_docs > 0:
        check_texts.extend(list(_iter_dataset_texts(args.dataset_check_docs)))
    strict = not args.no_strict
    report = _check_parity(
        rust_tok=rust_tok,
        hf_tok=hf_loaded,
        texts=check_texts,
        special_tokens=list(special_tokens_map.keys()),
        strict=strict,
    )

    print(f"source_dir: {source_dir}")
    print(f"output_dir: {output_dir}")
    print(f"vocab_size: {hf_loaded.vocab_size}")
    print(f"bos_token: {hf_loaded.bos_token} (id={hf_loaded.bos_token_id})")
    print(f"eos_token: {hf_loaded.eos_token} (id={hf_loaded.eos_token_id})")
    print(
        "parity: "
        f"checked={report.checked}, "
        f"id_mismatches={report.id_mismatches}, "
        f"decode_mismatches={report.decode_mismatches}, "
        f"special_id_mismatches={report.special_id_mismatches}"
    )
    print("done")


if __name__ == "__main__":
    main()
