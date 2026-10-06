"""
Preview tokenizer segmentation on a few sample strings.

Examples:
  python -m scripts.tok_preview
  python -m scripts.tok_preview --tokenizer-dir /path/to/tokenizer_superbpe
  python -m scripts.tok_preview --text "New York City" --text " the quick brown fox"
"""

import argparse
import os
from typing import List

from nanochat.tokenizer import get_tokenizer, _resolve_tokenizer_dir


DEFAULT_CASES = [
    "hello world",
    " the quick brown fox jumps over the lazy dog",
    "New York City is great.",
    "I went to the store and bought milk.",
    "  leading space,  multiple   spaces",
    "line one\nline two",
    "punctuation: (a), [b], {c}, and 12345.",
]


def _format_segment(segment: str) -> str:
    escaped = segment.encode("unicode_escape").decode("ascii")
    escaped = escaped.replace('"', '\\"')
    return f"\"{escaped}\""


def _token_segments(tokenizer, text: str, prepend_bos: bool) -> List[str]:
    prepend = "<|bos|>" if prepend_bos else None
    token_ids = tokenizer.encode(text, prepend=prepend)
    return [tokenizer.decode([tok]) for tok in token_ids]


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview tokenizer segmentation (no token IDs).")
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default="",
        help="Optional tokenizer directory override (same format as NANOCHAT_TOKENIZER_DIR).",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Test string to tokenize (can be provided multiple times).",
    )
    parser.add_argument(
        "--prepend-bos",
        action="store_true",
        help="Include BOS token at the beginning of each encoded example.",
    )
    args = parser.parse_args()

    if args.tokenizer_dir.strip():
        os.environ["NANOCHAT_TOKENIZER_DIR"] = os.path.abspath(os.path.expanduser(args.tokenizer_dir))

    tokenizer = get_tokenizer()
    tokenizer_dir = _resolve_tokenizer_dir()
    cases = args.text if args.text else DEFAULT_CASES

    print(f"Tokenizer dir: {tokenizer_dir}")
    print(f"Vocab size: {tokenizer.get_vocab_size():,}")
    print()

    for i, text in enumerate(cases, start=1):
        segments = _token_segments(tokenizer, text, prepend_bos=args.prepend_bos)
        joined = " | ".join(_format_segment(seg) for seg in segments)
        print(f"[{i}] input: {_format_segment(text)}")
        print(f"    segments ({len(segments)}): {joined}")
        print()


if __name__ == "__main__":
    main()
