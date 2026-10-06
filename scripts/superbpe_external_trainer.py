"""
External SuperBPE trainer.

Run this script with a Python environment that has tokenizers-superbpe fork
installed. It performs:
1) stage-1 BPE training with regular pretokenization
2) stage-2 extension from inherited merges.txt with SuperBPE regex
"""

import argparse
import json
import os
import tempfile
from contextlib import contextmanager
from typing import Dict, Iterable, List

from tokenizers import Regex, Tokenizer
from tokenizers import decoders, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


@contextmanager
def _temporary_cwd(path: str):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _build_tokenizer(pattern: str) -> Tokenizer:
    tokenizer = Tokenizer(BPE(
        byte_fallback=True,
        unk_token=None,
        fuse_unk=False,
    ))
    tokenizer.normalizer = None
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=Regex(pattern), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = None
    return tokenizer


def _build_trainer(vocab_size: int) -> BpeTrainer:
    return BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        min_frequency=0,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[],
    )


def _iter_text_jsonl(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            text = row.get("text")
            if not isinstance(text, str):
                raise TypeError(
                    f"Invalid row in {path} at line {line_no}: expected 'text' as string."
                )
            if text:
                yield text


def _read_merges_txt(path: str) -> List[str]:
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, raw in enumerate(f):
            line = raw.strip()
            if not line:
                continue
            if idx == 0 and line.startswith("#version"):
                continue
            lines.append(line)
    return lines


def _write_seed_merges(path: str, merges: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for merge in merges:
            f.write(f"{merge}\n")


def _count_docs(path: str) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _common_prefix_len(a: List[str], b: List[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _probe_superbpe_fork_behavior() -> Dict[str, object]:
    texts = [
        "hello world this is a tiny superbpe probe",
        "hello there world this is another tiny superbpe probe",
        "foobar baz qux hello world",
    ] * 30
    stage1_pattern = r"""\w+|[^\w\s]+|\s+"""
    stage2_pattern = r"""\s+|[^\s]+"""
    seed_count = 8

    with tempfile.TemporaryDirectory(prefix="superbpe_probe_") as probe_dir:
        with _temporary_cwd(probe_dir):
            stage1 = _build_tokenizer(stage1_pattern)
            stage1.train_from_iterator(texts, _build_trainer(vocab_size=320))
            stage1.model.save(probe_dir, "probe_stage1")
            stage1_merges = _read_merges_txt(os.path.join(probe_dir, "probe_stage1-merges.txt"))
            if len(stage1_merges) < seed_count:
                return {
                    "fork_probe_ok": False,
                    "probe_reason": "insufficient_probe_merges",
                    "probe_prefix_len": 0,
                    "probe_seed_count": seed_count,
                }

            seed = stage1_merges[:seed_count]
            _write_seed_merges(os.path.join(probe_dir, "merges.txt"), seed)

            stage2 = _build_tokenizer(stage2_pattern)
            stage2.train_from_iterator(texts, _build_trainer(vocab_size=340))
            stage2.model.save(probe_dir, "probe_stage2")
            stage2_merges = _read_merges_txt(os.path.join(probe_dir, "probe_stage2-merges.txt"))
            prefix_len = _common_prefix_len(seed, stage2_merges[:seed_count])
            return {
                "fork_probe_ok": bool(prefix_len == seed_count),
                "probe_reason": "ok" if prefix_len == seed_count else "prefix_mismatch",
                "probe_prefix_len": int(prefix_len),
                "probe_seed_count": seed_count,
            }


def main() -> None:
    parser = argparse.ArgumentParser(description="External SuperBPE trainer")
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--vocab-size-no-special", type=int, required=True)
    parser.add_argument("--transition-merges", type=int, required=True)
    parser.add_argument("--stage1-pattern", type=str, required=True)
    parser.add_argument("--stage2-pattern", type=str, required=True)
    args = parser.parse_args()

    input_jsonl = os.path.abspath(os.path.expanduser(args.input_jsonl))
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    vocab_size_no_special = int(args.vocab_size_no_special)
    transition_merges = int(args.transition_merges)
    if vocab_size_no_special < 256:
        raise ValueError(
            f"vocab_size_no_special must be >= 256, got {vocab_size_no_special}"
        )
    if transition_merges < 0:
        raise ValueError(f"transition_merges must be >= 0, got {transition_merges}")
    stage1_vocab_size = 256 + transition_merges
    if stage1_vocab_size > vocab_size_no_special:
        raise ValueError(
            f"stage1_vocab_size ({stage1_vocab_size}) > vocab_size_no_special ({vocab_size_no_special})"
        )

    docs = _count_docs(input_jsonl)
    if docs <= 0:
        raise RuntimeError(f"No docs found in input JSONL: {input_jsonl}")

    probe = _probe_superbpe_fork_behavior()
    if not bool(probe.get("fork_probe_ok", False)):
        raise RuntimeError(
            "SuperBPE fork probe failed before training. "
            f"reason={probe.get('probe_reason')} "
            f"prefix_len={probe.get('probe_prefix_len')} "
            f"seed_count={probe.get('probe_seed_count')}. "
            "This usually indicates stock tokenizers behavior (no merges.txt inheritance)."
        )

    with _temporary_cwd(output_dir):
        stage1 = _build_tokenizer(args.stage1_pattern)
        stage1.train_from_iterator(
            _iter_text_jsonl(input_jsonl),
            _build_trainer(vocab_size=stage1_vocab_size),
        )
        stage1.model.save(output_dir, "stage1")
        stage1_merges = _read_merges_txt(os.path.join(output_dir, "stage1-merges.txt"))
        if len(stage1_merges) < transition_merges:
            raise RuntimeError(
                f"Stage-1 produced only {len(stage1_merges)} merges, expected at least {transition_merges}."
            )

        _write_seed_merges(os.path.join(output_dir, "merges.txt"), stage1_merges[:transition_merges])

        stage2 = _build_tokenizer(args.stage2_pattern)
        stage2.train_from_iterator(
            _iter_text_jsonl(input_jsonl),
            _build_trainer(vocab_size=vocab_size_no_special),
        )
        stage2.model.save(output_dir, "final")
        final_merges = _read_merges_txt(os.path.join(output_dir, "final-merges.txt"))
        inherited_prefix_len = _common_prefix_len(
            stage1_merges[:transition_merges],
            final_merges[:transition_merges],
        )
        inheritance_prefix_ok = inherited_prefix_len == transition_merges

    manifest: Dict[str, object] = {
        "version": 1,
        "input_jsonl": input_jsonl,
        "output_dir": output_dir,
        "docs": docs,
        "vocab_size_no_special": vocab_size_no_special,
        "transition_merges": transition_merges,
        "stage1_vocab_size": stage1_vocab_size,
        "stage1_pattern": args.stage1_pattern,
        "stage2_pattern": args.stage2_pattern,
        "stage1_merges_count": len(stage1_merges),
        "final_merges_count": len(final_merges),
        "fork_probe_ok": bool(probe.get("fork_probe_ok", False)),
        "fork_probe_reason": str(probe.get("probe_reason", "")),
        "fork_probe_prefix_len": int(probe.get("probe_prefix_len", 0)),
        "fork_probe_seed_count": int(probe.get("probe_seed_count", 0)),
        "inherited_prefix_len": int(inherited_prefix_len),
        "inheritance_prefix_ok": bool(inheritance_prefix_ok),
    }
    manifest_path = os.path.join(output_dir, "superbpe_external_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
