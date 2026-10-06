"""
Build compositional runtime metadata for raw nanochat tokenizers.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]


SCRIPT_DIR = Path(__file__).resolve().parent
COBPE_REPO_ROOT = SCRIPT_DIR.parent
if str(COBPE_REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(COBPE_REPO_ROOT))

from transformers import PreTrainedTokenizerFast

from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer
from scripts.export_hf_tokenizer_from_rustbpe import _build_backend_tokenizer
from scripts.tok_train import (
    _export_rust_tokenizer_to_hf_dir,
    finalize_tokenizer_with_survivor_ids,
)
from cobpe.decomposition.prepositions import (
    DEFAULT_PREPOSITION_PROFILE,
    normalize_preposition_profile_name,
)


MULTI_TOKEN_FIRST_GROUPS = {
    "space_prefix",
    "base_capitalization",
    "determiners",
    "articles",
    "article_det",
    "article_space_prefix",
    "article_capitalization",
    "prepositions",
    "prep_space_prefix",
    "prep_capitalization",
    "prefix_punctuation",
}


def _decode(tokenizer, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except TypeError:
        try:
            return tokenizer.decode(token_ids, skip_special_tokens=False)
        except TypeError:
            return tokenizer.decode(token_ids)


MULTI_TOKEN_MODIFIER_PLACEMENTS = {"last", "first", "split_by_role"}
BUFFER_CONFIG_NAME = "tokenizer_buffer.json"
FINALIZE_STATS_NAME = "tokenizer_finalization.json"


def _normalize_multi_token_modifier_placement(value: str | None) -> str:
    key = str(value or "split_by_role").strip().lower()
    if key not in MULTI_TOKEN_MODIFIER_PLACEMENTS:
        valid = ", ".join(sorted(MULTI_TOKEN_MODIFIER_PLACEMENTS))
        raise ValueError(f"Unknown multi-token modifier placement {value!r}. Expected one of: {valid}")
    return key


def _save_hf_fast_from_rustbpe(source_tokenizer_dir: str, output_dir: str, *, overwrite: bool) -> None:
    from nanochat.tokenizer import RustBPETokenizer
    from cobpe.tokenization.fingerprints import fingerprint_named_files

    source_path = os.path.join(source_tokenizer_dir, "tokenizer.json")
    output_tokenizer_json = os.path.join(output_dir, "tokenizer.json")
    source_fingerprint = fingerprint_named_files(
        source_tokenizer_dir,
        ("tokenizer.pkl", "tokenizer.json"),
        required=("tokenizer.pkl",),
    )
    source_manifest_path = os.path.join(output_dir, "cobpe_source_fingerprint.json")
    if os.path.exists(output_dir):
        if not overwrite and os.path.exists(output_tokenizer_json) and os.path.exists(source_manifest_path):
            try:
                with open(source_manifest_path, "r", encoding="utf-8") as f:
                    cached_source_fingerprint = json.load(f)
            except (OSError, ValueError):
                cached_source_fingerprint = None
            if cached_source_fingerprint == source_fingerprint:
                return
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    rust_tokenizer = RustBPETokenizer.from_directory(source_tokenizer_dir)
    if os.path.exists(source_path):
        shutil.copy2(source_path, output_tokenizer_json)
    else:
        enc = rust_tokenizer.enc
        mergeable_ranks = getattr(enc, "_mergeable_ranks", None)
        pattern = getattr(enc, "_pat_str", None)
        special_tokens_map = getattr(enc, "_special_tokens", None)
        if not isinstance(mergeable_ranks, dict) or not pattern or not isinstance(special_tokens_map, dict):
            raise ValueError(
                "Unexpected tokenizer.pkl internals while exporting HF tokenizer; "
                "missing tiktoken mergeable ranks, pattern, or special tokens."
            )
        special_tokens_in_id_order = [t for t, _ in sorted(special_tokens_map.items(), key=lambda kv: kv[1])]
        backend = _build_backend_tokenizer(
            mergeable_ranks=mergeable_ranks,
            pattern=pattern,
            special_tokens_in_id_order=special_tokens_in_id_order,
        )
        backend.save(output_tokenizer_json)

    special_tokens = [tok for tok in SPECIAL_TOKENS if tok in rust_tokenizer.get_special_tokens()]
    extras = [tok for tok in rust_tokenizer.get_special_tokens() if tok not in special_tokens]
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=output_tokenizer_json,
        bos_token="<|bos|>",
        eos_token="<|bos|>",
        additional_special_tokens=special_tokens + extras,
    )
    hf_tok.save_pretrained(output_dir)
    for artifact_name in ("tokenizer.pkl", "tokenizer_buffer.json", "vd_buffer_config.json", "token_bytes.pt"):
        source_artifact = os.path.join(source_tokenizer_dir, artifact_name)
        if os.path.exists(source_artifact):
            shutil.copy2(source_artifact, os.path.join(output_dir, artifact_name))
    with open(source_manifest_path, "w", encoding="utf-8") as f:
        json.dump(source_fingerprint, f, indent=2, sort_keys=True)


def _load_buffer_config(tokenizer_dir: str) -> dict[str, Any] | None:
    path = os.path.join(tokenizer_dir, BUFFER_CONFIG_NAME)
    if not os.path.exists(path):
        path = os.path.join(tokenizer_dir, "vd_buffer_config.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid {BUFFER_CONFIG_NAME}: expected a JSON object")
    return payload


def _write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def _write_token_bytes(tokenizer: RustBPETokenizer, output_path: str) -> None:
    import torch

    special_set = set(tokenizer.get_special_tokens())
    lengths = []
    for token_id in range(tokenizer.get_vocab_size()):
        token_str = tokenizer.decode([token_id])
        lengths.append(0 if token_str in special_set else len(token_str.encode("utf-8")))
    torch.save(torch.tensor(lengths, dtype=torch.int32), output_path)


def _prepare_finalized_tokenizer(
    *,
    tokenizer_dir: str,
    staging_dir: str,
    buffer_config: dict[str, Any],
    survivor_ids,
) -> tuple[RustBPETokenizer, dict[str, Any], dict[str, Any]]:
    target_vocab_size = int(buffer_config.get("vocab_size_target", -1))
    train_vocab_size = int(buffer_config.get("vocab_size_train", -1))
    if target_vocab_size <= 0 or train_vocab_size <= target_vocab_size:
        raise ValueError(
            f"Invalid buffered tokenizer sizes: target={target_vocab_size}, train={train_vocab_size}"
        )
    tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)
    if tokenizer.get_vocab_size() != train_vocab_size:
        raise ValueError(
            "Unfinalized buffered tokenizer size mismatch: "
            f"artifact={tokenizer.get_vocab_size()} metadata={train_vocab_size}"
        )
    finalized, stats = finalize_tokenizer_with_survivor_ids(
        tokenizer,
        survivor_ids=survivor_ids,
        target_vocab_size=target_vocab_size,
        train_vocab_size=train_vocab_size,
    )
    if finalized.get_vocab_size() != target_vocab_size:
        raise RuntimeError(
            f"Buffered tokenizer finalization produced {finalized.get_vocab_size()} tokens; "
            f"expected {target_vocab_size}"
        )

    finalized.save(staging_dir)
    _export_rust_tokenizer_to_hf_dir(finalized, staging_dir)
    _write_token_bytes(finalized, os.path.join(staging_dir, "token_bytes.pt"))
    finalized_config = dict(buffer_config)
    finalized_config.update(
        {
            "finalized_at_metadata_export": True,
            "runtime_vocab_size": target_vocab_size,
        }
    )
    finalized_config.pop("finalized_at_tok_train", None)
    _write_json(os.path.join(staging_dir, BUFFER_CONFIG_NAME), finalized_config)
    _write_json(os.path.join(staging_dir, FINALIZE_STATS_NAME), stats)
    return finalized, finalized_config, stats


def _commit_finalized_tokenizer(staging_dir: str, tokenizer_dir: str) -> None:
    for artifact_name in (
        "tokenizer.pkl",
        "tokenizer.json",
        "token_bytes.pt",
        BUFFER_CONFIG_NAME,
        FINALIZE_STATS_NAME,
    ):
        source = os.path.join(staging_dir, artifact_name)
        if not os.path.exists(source):
            raise FileNotFoundError(f"Finalized tokenizer staging artifact is missing: {source}")
    for artifact_name in (
        "tokenizer.pkl",
        "tokenizer.json",
        "token_bytes.pt",
        BUFFER_CONFIG_NAME,
        FINALIZE_STATS_NAME,
    ):
        os.replace(
            os.path.join(staging_dir, artifact_name),
            os.path.join(tokenizer_dir, artifact_name),
        )


def _build_bundle_args(base_tokenizer_path: str, args: argparse.Namespace) -> SimpleNamespace:
    """Return the supported paper-surface settings consumed by the bundle builder."""
    return SimpleNamespace(
        base_tokenizer=base_tokenizer_path,
        language=args.language,
        tokenization_mode="dual_stream",
        skip_multi_token_words=False,
        skip_three_token_words=False,
        skip_four_token_words=False,
        skip_multi_token_bases=False,
        dont_decompose_spaces=False,
        skip_if_no_space_prefix=True,
        allow_no_space_prefix=False,
        dont_decompose_capitalized=False,
        enable_all_caps_capitalization=False,
        use_relative_space_cap_transforms=bool(args.use_relative_space_cap_transforms),
        canonicalize_token_strings=True,
        use_type_str_as_type=False,
        skip_stop_words=False,
        filter_propernouns_and_aux=False,
        merge_plural_and_present_singular=False,
        include_non_resource_latin_tokens=True,
        include_non_latin_tokens=True,
        decompose_punctuation=bool(args.decompose_punctuation),
        decompose_articles=bool(args.decompose_articles),
        decompose_possessive_determiners=bool(args.decompose_possessive_determiners),
        decompose_demonstrative_determiners=bool(args.decompose_demonstrative_determiners),
        decompose_quantifier_determiners=bool(args.decompose_quantifier_determiners),
        decompose_prepositions=bool(args.decompose_prepositions),
        decompose_article_prep_space_prefix=False,
        preposition_profile=str(args.preposition_profile),
        preposition_list=list(args.preposition_list) if args.preposition_list else None,
        modifier_generation_mode="vocab_only",
        multitoken_allowed_groups=[],
        word_boundary_safety=bool(args.word_boundary_safety),
        multi_token_modifier_position=str(args.multi_token_modifier_position),
        prune_punct_tokens=False,
        possessive_separate_group=bool(args.possessive_separate_group),
        overwrite_bundle_cache=bool(args.overwrite_metadata_cache),
        reuse_decomposition_stage_cache=True,
        decomposition_stage_cache_dir=None,
        overwrite_decomposition_stage_cache=bool(args.overwrite_metadata_cache),
        enable_build_optimizations=True,
        max_vocab_items=None,
        max_tokens_per_base_word=None,
        log_second_pass_bases=False,
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        text_field="text",
        output_dir=args.output_dir,
        tokenizer_cache_dir=args.metadata_cache_dir,
        use_local_parquet=True,
        local_parquet_dir=args.local_parquet_dir,
        auto_local_nanochat=False,
    )


def _spread_multi_token_modifiers(combined_modifier: list[int], groups: list[str], base_len: int, position: str) -> list[list[int]]:
    position = _normalize_multi_token_modifier_placement(position)
    if base_len <= 1:
        return [list(combined_modifier)]
    empty = [0] * len(combined_modifier)
    if position == "last":
        return [list(empty) for _ in range(base_len - 1)] + [list(combined_modifier)]
    if position == "first":
        return [list(combined_modifier)] + [list(empty) for _ in range(base_len - 1)]
    first_modifier = [0] * len(combined_modifier)
    last_modifier = [0] * len(combined_modifier)
    for idx, group_name in enumerate(groups):
        value = int(combined_modifier[idx])
        if value <= 0:
            continue
        if group_name in MULTI_TOKEN_FIRST_GROUPS:
            first_modifier[idx] = value
        else:
            last_modifier[idx] = value
    if base_len == 2:
        return [first_modifier, last_modifier]
    return [first_modifier] + [[0] * len(combined_modifier) for _ in range(base_len - 2)] + [last_modifier]


def _default_indices_for_group(names: list[str]) -> list[int]:
    defaults = []
    for idx, name in enumerate(names):
        lower = str(name).lower()
        if lower.startswith("no_") or lower.startswith("na_") or lower == "none":
            defaults.append(idx)
    return defaults or [0]


def _build_group_value_names(bundle, groups: list[str]) -> tuple[dict[str, list[str]], dict[str, list[int]]]:
    model_init_data = getattr(bundle, "model_init_data", {}) or {}
    types_loss_indices_map = model_init_data.get("types_loss_indices_map", {}) or {}
    transformation_names_to_int = model_init_data.get("transformation_names_to_int", {}) or {}
    int_to_name = {int(v): str(k) for k, v in transformation_names_to_int.items()}

    group_value_names: dict[str, list[str]] = {}
    default_indices: dict[str, list[int]] = {}
    for group in groups:
        bounds = types_loss_indices_map.get(group)
        if bounds is None:
            names: list[str] = []
        else:
            start, end = int(bounds[0]), int(bounds[1])
            names = [int_to_name.get(idx, f"idx_{idx - start}") for idx in range(start, end)]
        group_value_names[group] = names
        default_indices[group] = _default_indices_for_group(names)
    return group_value_names, default_indices


def _build_payload(bundle, *, multi_token_modifier_position: str) -> tuple[dict[str, Any], dict[str, Any]]:
    modifier_array = bundle.unified_modifier_array
    if modifier_array is None:
        raise ValueError("Bundle is missing unified_modifier_array.")
    if bundle.sequence_map is None:
        raise ValueError("Bundle is missing sequence_map.")

    seq_dict = bundle.sequence_map.to_dict()
    groups = list(modifier_array.groups)
    group_sizes = [int(modifier_array.group_sizes[group]) for group in groups]
    default_modifier = [0] * len(groups)
    group_value_names, default_indices = _build_group_value_names(bundle, groups)
    entries = []
    inverse_entries: dict[tuple[int, tuple[int, ...]], str] = {}
    collisions = 0
    for entry in seq_dict.get("sequences", []):
        token_ids = [int(v) for v in entry.get("token_ids", [])]
        base_ids = [int(v) for v in entry.get("base_ids", [])]
        modifier = [int(v) for v in entry.get("modifier", default_modifier)]
        if not token_ids or not base_ids:
            continue
        modifier_rows = _spread_multi_token_modifiers(modifier, groups, len(base_ids), multi_token_modifier_position)
        surface = _decode(bundle.tokenizer, token_ids)
        entries.append(
            {
                "token_ids": token_ids,
                "base_ids": base_ids,
                "modifier_rows": modifier_rows,
                "surface": surface,
            }
        )
        if len(base_ids) == 1 and len(modifier_rows) == 1:
            key = (int(base_ids[0]), tuple(int(v) for v in modifier_rows[0]))
            existing = inverse_entries.get(key)
            if existing is not None and existing != surface:
                collisions += 1
            else:
                inverse_entries[key] = surface

    payload = {
        "version": 1,
        "group_names": groups,
        "group_value_names": group_value_names,
        "default_indices": default_indices,
        "num_modifier_groups": len(groups),
        "modifier_group_sizes": group_sizes,
        "default_modifier": default_modifier,
        "entries": entries,
        "inverse_entries": [
            {
                "base_id": base_id,
                "modifier": list(modifier_row),
                "surface": surface,
            }
            for (base_id, modifier_row), surface in sorted(inverse_entries.items())
        ],
    }
    report = {
        "num_entries": len(entries),
        "num_inverse_entries": len(inverse_entries),
        "inverse_collisions_skipped": collisions,
        "modifier_group_sizes": group_sizes,
        "groups": groups,
        "group_value_names_present": {group: len(names) for group, names in group_value_names.items()},
        "multi_token_modifier_position": multi_token_modifier_position,
    }
    return payload, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export raw compositional metadata for a tokenizer.")
    parser.add_argument("--base-dir", type=str, default="", help="Override NANOCHAT_BASE_DIR.")
    parser.add_argument("--tokenizer-dir", type=str, default="", help="Tokenizer dir containing tokenizer.pkl/json.")
    parser.add_argument("--hf-tokenizer-dir", type=str, default="", help="Optional HF export dir used while building metadata.")
    parser.add_argument("--metadata-cache-dir", type=str, default="", help="Metadata/bundle build cache dir.")
    parser.add_argument("--output-path", type=str, default="", help="Where to write compositional.json.")
    parser.add_argument("--report-path", type=str, default="", help="Optional export report JSON path.")
    parser.add_argument("--output-dir", type=str, default="", help="Optional output dir for metadata args/cache key.")
    parser.add_argument("--dataset", type=str, default=os.environ.get("DATASET", "karpathy/climbmix-400b-shuffle"))
    parser.add_argument("--dataset-config", type=str, default=os.environ.get("DATASET_CONFIG", "default"))
    parser.add_argument("--local-parquet-dir", type=str, default=os.environ.get("LOCAL_PARQUET_DIR", ""))
    parser.add_argument("--language", type=str, default=os.environ.get("BUNDLE_LANGUAGE", "en"))
    parser.add_argument("--use-relative-space-cap-transforms", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--preposition-profile",
        type=str,
        default=os.environ.get("PREPOSITION_PROFILE", DEFAULT_PREPOSITION_PROFILE),
        choices=["core", "expanded", "full"],
        help="Named preposition profile: core, expanded, or full.",
    )
    parser.add_argument("--preposition-list", nargs="*", default=None)
    parser.add_argument(
        "--multi-token-modifier-placement",
        dest="multi_token_modifier_position",
        type=str,
        default="split_by_role",
        choices=["last", "first", "split_by_role"],
        help=(
            "Where to place span-level modifiers over multi-token bases: last, first, or split_by_role."
        ),
    )
    parser.add_argument("--decompose-articles", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--decompose-possessive-determiners", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--decompose-demonstrative-determiners", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--decompose-quantifier-determiners", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--decompose-prepositions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--decompose-punctuation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--word-boundary-safety", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--possessive-separate-group", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite HF export dir and compositional.json.")
    parser.add_argument("--overwrite-metadata-cache", action="store_true", help="Rebuild the cached metadata inputs.")
    args = parser.parse_args()

    from cobpe.decomposition.bundle import (
        get_or_create_tokenizer_bundle,
        get_tokenizer_cache_path,
    )

    if args.base_dir.strip():
        os.environ["NANOCHAT_BASE_DIR"] = os.path.abspath(os.path.expanduser(args.base_dir))

    base_dir = get_base_dir()
    tokenizer_dir = args.tokenizer_dir.strip() or os.path.join(base_dir, "tokenizer")
    tokenizer_dir = os.path.realpath(os.path.abspath(os.path.expanduser(tokenizer_dir)))
    tokenizer_variant_name = os.path.basename(tokenizer_dir.rstrip(os.sep)) or "tokenizer"
    hf_tokenizer_dir = args.hf_tokenizer_dir.strip() or f"{tokenizer_dir}_hf"
    hf_tokenizer_dir = os.path.abspath(os.path.expanduser(hf_tokenizer_dir))
    args.preposition_profile = normalize_preposition_profile_name(args.preposition_profile)
    args.multi_token_modifier_position = _normalize_multi_token_modifier_placement(args.multi_token_modifier_position)

    tokenizer_cache_dir = (
        args.metadata_cache_dir.strip()
        or os.environ.get("TOKENIZER_CACHE_DIR", "").strip()
        or os.path.join(base_dir, "tokenizer_cache")
    )
    tokenizer_cache_dir = os.path.abspath(os.path.expanduser(tokenizer_cache_dir))
    output_path = args.output_path.strip() or os.path.join(tokenizer_dir, "compositional.json")
    output_path = os.path.abspath(os.path.expanduser(output_path))
    report_path = args.report_path.strip() or os.path.join(tokenizer_dir, "compositional_export_report.json")
    report_path = os.path.abspath(os.path.expanduser(report_path))

    if not os.path.exists(os.path.join(tokenizer_dir, "tokenizer.pkl")):
        raise FileNotFoundError(f"tokenizer.pkl not found in {tokenizer_dir}")
    if os.path.exists(output_path) and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")

    os.makedirs(tokenizer_cache_dir, exist_ok=True)
    buffer_config = _load_buffer_config(tokenizer_dir)
    source_tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)
    finalization_required = False
    target_vocab_size = source_tokenizer.get_vocab_size()
    if buffer_config is not None:
        target_vocab_size = int(buffer_config.get("vocab_size_target", -1))
        train_vocab_size = int(buffer_config.get("vocab_size_train", -1))
        finalized = bool(buffer_config.get("finalized_at_metadata_export", False))
        current_vocab_size = source_tokenizer.get_vocab_size()
        if target_vocab_size <= 0 or train_vocab_size < target_vocab_size:
            raise ValueError(
                f"Invalid {BUFFER_CONFIG_NAME} sizes: target={target_vocab_size}, train={train_vocab_size}"
            )
        if train_vocab_size > target_vocab_size:
            if current_vocab_size == train_vocab_size and not finalized:
                finalization_required = True
            elif current_vocab_size == target_vocab_size and finalized:
                finalization_required = False
            elif current_vocab_size == target_vocab_size and buffer_config.get("finalized_at_tok_train", False):
                raise RuntimeError(
                    "This tokenizer was compacted by the deprecated tokenizer-training finalizer, "
                    "before the exact CoBPE decomposition was known. Retrain it with the buffer and "
                    "run scripts.export_compositional_metadata."
                )
            else:
                raise RuntimeError(
                    "Buffered CoBPE tokenizer is in an inconsistent finalization state: "
                    f"current={current_vocab_size}, target={target_vocab_size}, train={train_vocab_size}, "
                    f"finalized={finalized}. Rebuild the tokenizer artifacts."
                )

    # First pass: build the exact requested CoBPE decomposition over the buffered
    # tokenizer. Its non_inflection_indices are the authoritative survivor set.
    _save_hf_fast_from_rustbpe(tokenizer_dir, hf_tokenizer_dir, overwrite=args.overwrite)
    bundle_args = _build_bundle_args(hf_tokenizer_dir, args)
    bundle_args.tokenizer_cache_dir = tokenizer_cache_dir
    bundle = get_or_create_tokenizer_bundle(bundle_args, cache_base_dir=tokenizer_cache_dir)
    cache_path = get_tokenizer_cache_path(bundle_args, tokenizer_cache_dir)
    staging_temp = None
    finalization_stats = None

    if finalization_required:
        survivor_ids = bundle.model_init_data.get("non_inflection_indices")
        if not survivor_ids:
            raise RuntimeError("Buffered CoBPE finalization found no survivor IDs in the decomposition bundle")
        staging_temp = tempfile.TemporaryDirectory(
            prefix="cobpe_finalized_",
            dir=os.path.dirname(tokenizer_dir),
        )
        staging_dir = staging_temp.name
        finalized_tokenizer, _finalized_config, finalization_stats = _prepare_finalized_tokenizer(
            tokenizer_dir=tokenizer_dir,
            staging_dir=staging_dir,
            buffer_config=buffer_config,
            survivor_ids=survivor_ids,
        )

        # Second pass: all IDs and decomposition entries must be rebuilt against
        # the compact tokenizer. The cache key includes tokenizer contents.
        _save_hf_fast_from_rustbpe(staging_dir, hf_tokenizer_dir, overwrite=True)
        bundle_args = _build_bundle_args(hf_tokenizer_dir, args)
        bundle_args.tokenizer_cache_dir = tokenizer_cache_dir
        bundle = get_or_create_tokenizer_bundle(bundle_args, cache_base_dir=tokenizer_cache_dir)
        cache_path = get_tokenizer_cache_path(bundle_args, tokenizer_cache_dir)
        compact_base_size = len(bundle.model_init_data.get("non_inflection_indices", []))
        if compact_base_size != target_vocab_size:
            raise RuntimeError(
                "CoBPE finalization is not closed under the requested decomposition: "
                f"final tokenizer has {target_vocab_size} tokens but second-pass survivor set has "
                f"{compact_base_size}. Refusing to publish inconsistent runtime artifacts."
            )
        if finalized_tokenizer.get_vocab_size() != target_vocab_size:
            raise RuntimeError("Finalized tokenizer changed size unexpectedly during metadata rebuild")

    payload, report = _build_payload(
        bundle,
        multi_token_modifier_position=bundle_args.multi_token_modifier_position,
    )
    payload["source"] = {
        "tokenizer_dir": tokenizer_dir,
        "hf_tokenizer_dir": hf_tokenizer_dir,
        "bundle_cache_path": cache_path,
        "variant_name": tokenizer_variant_name,
    }
    payload["build_config"] = {
        "decompose_articles": bool(bundle_args.decompose_articles),
        "decompose_possessive_determiners": bool(bundle_args.decompose_possessive_determiners),
        "decompose_demonstrative_determiners": bool(bundle_args.decompose_demonstrative_determiners),
        "decompose_quantifier_determiners": bool(bundle_args.decompose_quantifier_determiners),
        "decompose_prepositions": bool(bundle_args.decompose_prepositions),
        "use_relative_space_cap_transforms": bool(bundle_args.use_relative_space_cap_transforms),
        "decompose_punctuation": bool(bundle_args.decompose_punctuation),
        "word_boundary_safety": bool(bundle_args.word_boundary_safety),
        "multi_token_modifier_position": str(bundle_args.multi_token_modifier_position),
        "preposition_profile": str(bundle_args.preposition_profile),
        "preposition_list": list(bundle_args.preposition_list) if bundle_args.preposition_list else None,
    }
    payload["runtime_vocab_size"] = int(target_vocab_size)
    if finalization_stats is not None:
        payload["vocab_finalization"] = dict(finalization_stats)

    all_runtime_ids = [
        int(token_id)
        for entry in payload.get("entries", [])
        for field in ("token_ids", "base_ids")
        for token_id in entry.get(field, [])
    ]
    all_runtime_ids.extend(
        int(entry["base_id"])
        for entry in payload.get("inverse_entries", [])
        if "base_id" in entry
    )
    if all_runtime_ids and (min(all_runtime_ids) < 0 or max(all_runtime_ids) >= target_vocab_size):
        raise RuntimeError(
            "Compositional metadata contains IDs outside the finalized runtime vocabulary: "
            f"min={min(all_runtime_ids)} max={max(all_runtime_ids)} vocab={target_vocab_size}"
        )

    if staging_temp is not None:
        _commit_finalized_tokenizer(staging_temp.name, tokenizer_dir)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    report.update(
        {
            "tokenizer_dir": tokenizer_dir,
            "hf_tokenizer_dir": hf_tokenizer_dir,
            "bundle_cache_path": cache_path,
            "output_path": output_path,
            "report_path": report_path,
            "runtime_vocab_size": int(target_vocab_size),
            "vocab_finalization": dict(finalization_stats) if finalization_stats is not None else None,
        }
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    if staging_temp is not None:
        staging_temp.cleanup()

    print(f"Wrote compositional metadata to {output_path}")
    print(f"Wrote export report to {report_path}")
    print(f"HF tokenizer dir: {hf_tokenizer_dir}")
    print(f"Bundle cache path: {cache_path}")


if __name__ == "__main__":
    main()
