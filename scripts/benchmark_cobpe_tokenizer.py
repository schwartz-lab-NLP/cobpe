"""Benchmark the Rust CoBPE batch path with repeatable inputs.

The benchmark intentionally calls ``rust_backend.process_text_batch`` directly:
this is the parallel Rust path used by ``encode_with_modifiers(list[str])`` before
the inexpensive BOS/append handling in Python.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Iterable


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="current", help="revision/build label in JSON output")
    parser.add_argument("--tokenizer-dir", default=os.environ.get("NANOCHAT_TOKENIZER_DIR", ""))
    parser.add_argument("--text-file", action="append", type=Path, default=[])
    parser.add_argument("--local-parquet-dir", default="")
    parser.add_argument("--parquet-index", type=int, default=0)
    parser.add_argument("--row-group", type=int, default=0)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--end-row", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--rayon-threads", type=int, default=1)
    parser.add_argument(
        "--no-synthetic",
        action="store_true",
        help="benchmark only supplied text files or parquet rows",
    )
    parser.add_argument(
        "--disable-gc",
        action="store_true",
        help="exclude Python cyclic-GC scan jitter from timed calls",
    )
    return parser


def _chunks(documents: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(documents), size):
        yield documents[start : start + size]


def _load_parquet(args: argparse.Namespace) -> list[str]:
    if not args.local_parquet_dir:
        return []
    import pyarrow.parquet as pq

    from nanochat.dataset import list_parquet_files

    paths = list_parquet_files(data_dir=args.local_parquet_dir)
    if not 0 <= args.parquet_index < len(paths):
        raise ValueError(
            f"parquet index {args.parquet_index} is out of range for {len(paths)} files"
        )
    parquet = pq.ParquetFile(paths[args.parquet_index])
    table = parquet.read_row_group(args.row_group, columns=["text"])
    documents = table.column("text").to_pylist()[args.start_row : args.end_row]
    if not all(isinstance(document, str) for document in documents):
        raise ValueError("selected parquet range contains null or non-string text")
    return documents


def _synthetic_cases() -> dict[str, list[str]]:
    ordinary = (
        "The quick brown fox jumps over the lazy dog. This is ordinary English "
        "prose with punctuation, articles, and prepositions.\n"
    ) * 400
    non_ascii = (
        "A café résumé mentions naïve coworkers, an em dash — and “quoted” "
        "English text.\n"
    ) * 400
    base64 = "QWxhZGRpbjpvcGVuIHNlc2FtZQ==" * 1600
    byte_fallback = "🙂🚀🧬🫠 हिन्दी भाषा परीक्षण 中文测试𐍈 " * 1400
    ordinary_batch = [ordinary[start : start + 2048] for start in range(0, len(ordinary), 2048)]
    return {
        "ordinary_english": [ordinary],
        "non_ascii_english": [non_ascii],
        "base64_heavy": [base64],
        "byte_fallback_heavy": [byte_fallback],
        "ordinary_batch": ordinary_batch * 4,
    }


def _time_case(backend, documents: list[str], warmups: int, repetitions: int) -> dict[str, object]:
    for _ in range(warmups):
        backend.process_text_batch(documents)
    elapsed = []
    token_count = 0
    for _ in range(repetitions):
        started = time.perf_counter()
        encoded = backend.process_text_batch(documents)
        elapsed.append(time.perf_counter() - started)
        token_count = sum(len(token_ids) for token_ids, _ in encoded)
    median = statistics.median(elapsed)
    return {
        "documents": len(documents),
        "characters": sum(map(len, documents)),
        "tokens": token_count,
        "median_seconds": median,
        "minimum_seconds": min(elapsed),
        "characters_per_second": sum(map(len, documents)) / median,
        "tokens_per_second": token_count / median,
        "runs_seconds": elapsed,
    }


def main() -> int:
    args = _parser().parse_args()
    if args.batch_size <= 0 or args.warmups < 0 or args.repetitions <= 0:
        raise SystemExit("batch size and repetitions must be positive; warmups cannot be negative")
    if args.rayon_threads <= 0:
        raise SystemExit("--rayon-threads must be positive")
    if args.tokenizer_dir:
        os.environ["NANOCHAT_TOKENIZER_DIR"] = str(Path(args.tokenizer_dir).resolve())
    os.environ["RAYON_NUM_THREADS"] = str(args.rayon_threads)

    from nanochat.tokenizer import get_tokenizer

    tokenizer = get_tokenizer()
    if not getattr(tokenizer, "_use_rust_backend", False):
        raise RuntimeError("benchmark requires the active Rust compositional backend")
    backend = tokenizer.rust_backend
    print(
        json.dumps(
            {
                "type": "environment",
                "label": args.label,
                "rust_backend": backend.build_info,
                "rayon_threads": args.rayon_threads,
                "gc_enabled_during_timing": not args.disable_gc,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    cases = {} if args.no_synthetic else _synthetic_cases()
    for path in args.text_file:
        text = path.read_text(encoding="utf-8")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cases[f"file:{path.name}:{digest[:12]}"] = [text]
    parquet_documents = _load_parquet(args)
    for batch_index, documents in enumerate(_chunks(parquet_documents, args.batch_size)):
        cases[f"parquet_batch:{batch_index}"] = documents
    if not cases:
        raise SystemExit("no benchmark cases selected")

    if args.disable_gc:
        gc.collect()
        gc.disable()
    try:
        for name, documents in cases.items():
            result = _time_case(backend, documents, args.warmups, args.repetitions)
            result.update({"type": "result", "label": args.label, "case": name})
            print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        if args.disable_gc:
            gc.enable()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
