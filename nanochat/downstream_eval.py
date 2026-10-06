import json
import math
import os
import random
import re
import string
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from tqdm import tqdm

from nanochat.common import print0
from nanochat.core_eval import (
    normalize_qa_answer_squad_style,
)
from nanochat.execution import execute_code


@dataclass(frozen=True)
class DownstreamTaskSpec:
    label: str
    category: str
    metric: str  # one of: mc | em | pass@10
    source_kind: str  # one of: bundle | arithmetic | cute | drop | hotpot | mbpp | humaneval
    bundle_dataset_uri: str = ""
    bundle_task_type: str = ""
    continuation_delimiter: str = " "
    num_fewshot: int = 5
    qa_style_normalization: bool = False


@dataclass
class DownstreamEvalConfig:
    max_per_task: int = 5000
    sample_seed: int = 1337
    verbose: bool = False
    log_every: int = 100
    em_batch_size: int = 8
    coding_batch_size: int = 2

    em_max_new_tokens: int = 64
    em_temperature: float = 0.0
    em_top_p: float = 1.0
    em_top_k: int = 0

    coding_num_samples: int = 20
    coding_pass_k: int = 10
    coding_temperature: float = 0.8
    coding_top_p: float = 0.95
    coding_top_k: int = 0
    coding_max_new_tokens: int = 128
    suite: str = "downstream"
    task_filter: str = "all"
    task_exclude_filter: str = ""
    preset: str = "default"


GenerateFn = Callable[..., List[str]]
GenerateBatchFn = Callable[..., List[List[str]]]
MCScoreFn = Callable[[List[dict], Dict[str, object]], float]
GenerationTimingFn = Callable[[str], Optional[Dict[str, float]]]
EMDebugHook = Callable[[Dict[str, object]], None]
TaskResultCallback = Callable[[Dict[str, object]], None]


_GSM_RE = re.compile(r"####\s*(\-?[0-9\.,]+)")
_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)\n```", re.DOTALL)
_FIRST_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
EXPECTED_DOWNSTREAM_TASK_COUNT = 30
_QA_STYLE_TASKS = {"TriviaQA", "CoQA", "DROP", "HotpotQA", "SQuAD"}


def _default_specs() -> List[DownstreamTaskSpec]:
    return [
        # Knowledge
        DownstreamTaskSpec("ARC-Easy", "Knowledge", "mc", "bundle", "world_knowledge/arc_easy.jsonl", "multiple_choice", "\nAnswer: ", 5),
        DownstreamTaskSpec("ARC-Challenge", "Knowledge", "mc", "bundle", "world_knowledge/arc_challenge.jsonl", "multiple_choice", "\nAnswer: ", 5),
        DownstreamTaskSpec("Jeopardy", "Knowledge", "em", "bundle", "world_knowledge/jeopardy_all.jsonl", "language_modeling", "\nAnswer: ", 5),
        DownstreamTaskSpec("MMLU", "Knowledge", "mc", "bundle", "world_knowledge/mmlu.jsonl", "multiple_choice", "\nAnswer: ", 5),
        DownstreamTaskSpec("OpenbookQA", "Knowledge", "mc", "bundle", "commonsense_reasoning/openbook_qa.jsonl", "multiple_choice", " ", 5),
        DownstreamTaskSpec("TriviaQA", "Knowledge", "em", "bundle", "world_knowledge/triviaqa.jsonl", "language_modeling", " ", 5, True),
        DownstreamTaskSpec("WikidataQA", "Knowledge", "em", "bundle", "world_knowledge/bigbench_qa_wikidata.jsonl", "language_modeling", " ", 5),

        # Math & Reasoning
        DownstreamTaskSpec("Arithmetic", "Math & Reasoning", "em", "arithmetic", num_fewshot=5),
        DownstreamTaskSpec("GSM8K", "Math & Reasoning", "em", "bundle", "symbolic_problem_solving/gsm8k.jsonl", "language_modeling", " ", 5),
        DownstreamTaskSpec("LSAT-AR", "Math & Reasoning", "mc", "bundle", "symbolic_problem_solving/agi_eval_lsat_ar.jsonl", "multiple_choice", " ", 5),
        DownstreamTaskSpec("Operators", "Math & Reasoning", "em", "bundle", "symbolic_problem_solving/bigbench_operators.jsonl", "language_modeling", " ", 5),
        DownstreamTaskSpec("Repeat-Copy-Logic", "Math & Reasoning", "em", "bundle", "symbolic_problem_solving/bigbench_repeat_copy_logic.jsonl", "language_modeling", " ", 5),

        # Coding
        DownstreamTaskSpec("HumanEval", "Coding", "pass@10", "humaneval", num_fewshot=0),
        DownstreamTaskSpec("MBPP", "Coding", "pass@10", "mbpp", num_fewshot=0),

        # Reading Comprehension
        DownstreamTaskSpec("BoolQ", "Reading Comprehension", "mc", "bundle", "reading_comprehension/boolq.jsonl", "multiple_choice", "\nAnswer: ", 5),
        DownstreamTaskSpec("CoQA", "Reading Comprehension", "em", "bundle", "reading_comprehension/coqa.jsonl", "language_modeling", " ", 0, True),
        DownstreamTaskSpec("DROP", "Reading Comprehension", "em", "drop", num_fewshot=5, qa_style_normalization=True),
        DownstreamTaskSpec("HotpotQA", "Reading Comprehension", "em", "hotpot", num_fewshot=5, qa_style_normalization=True),
        DownstreamTaskSpec("SQuAD", "Reading Comprehension", "em", "bundle", "reading_comprehension/squad.jsonl", "language_modeling", " ", 5, True),

        # Commonsense
        DownstreamTaskSpec("CommonsenseQA", "Commonsense", "mc", "bundle", "commonsense_reasoning/commonsense_qa.jsonl", "multiple_choice", " ", 5),
        DownstreamTaskSpec("COPA", "Commonsense", "mc", "bundle", "commonsense_reasoning/copa.jsonl", "multiple_choice", " ", 5),
        DownstreamTaskSpec("PIQA", "Commonsense", "mc", "bundle", "commonsense_reasoning/piqa.jsonl", "multiple_choice", "\nAnswer: ", 5),
        DownstreamTaskSpec("Winograd", "Commonsense", "mc", "bundle", "language_understanding/winograd_wsc.jsonl", "schema", " ", 5),
        DownstreamTaskSpec("Winogrande", "Commonsense", "mc", "bundle", "language_understanding/winogrande.jsonl", "schema", " ", 5),

        # Language Understanding
        DownstreamTaskSpec("HellaSwag", "Language Understanding", "mc", "bundle", "language_understanding/hellaswag.jsonl", "multiple_choice", " ", 5),
        DownstreamTaskSpec("LAMBADA", "Language Understanding", "em", "bundle", "language_understanding/lambada_openai.jsonl", "language_modeling", " ", 5),
        DownstreamTaskSpec("Language Identification", "Language Understanding", "em", "bundle", "language_understanding/bigbench_language_identification.jsonl", "multiple_choice", "\nAnswer: ", 5),

        # String Manipulation
        DownstreamTaskSpec("CS Algorithms", "String Manipulation", "em", "bundle", "symbolic_problem_solving/bigbench_cs_algorithms.jsonl", "language_modeling", " ", 5),
        DownstreamTaskSpec("CUTE", "String Manipulation", "em", "cute", num_fewshot=5),
        DownstreamTaskSpec("Dyck-Languages", "String Manipulation", "em", "bundle", "symbolic_problem_solving/bigbench_dyck_languages.jsonl", "language_modeling", " ", 5),
    ]


def _resolve_suite_alias(suite: str) -> str:
    key = str(suite or "downstream").strip().lower().replace("-", "_")
    if key in {"downstream", "all"}:
        return "downstream"
    if key in {"downstream_lm", "lm"}:
        return "downstream_lm"
    raise ValueError(f"Unsupported downstream suite: {suite!r} (expected downstream or downstream_lm)")


def _select_specs_for_suite(specs: Sequence[DownstreamTaskSpec], suite: str) -> List[DownstreamTaskSpec]:
    suite_key = _resolve_suite_alias(suite)
    if suite_key == "downstream":
        return list(specs)
    if suite_key == "downstream_lm":
        return [s for s in specs if s.metric == "em"]
    raise RuntimeError(f"Unhandled downstream suite selector: {suite_key}")


def get_downstream_specs(suite: str = "downstream") -> Tuple[DownstreamTaskSpec, ...]:
    all_specs = _default_specs()
    selected = _select_specs_for_suite(all_specs, suite)
    return tuple(selected)


def _normalize_task_label(label: str) -> str:
    return str(label or "").strip().lower()


def _filter_specs_for_task_filter(
    specs: Sequence[DownstreamTaskSpec],
    task_filter: str,
    suite: str,
) -> List[DownstreamTaskSpec]:
    raw = str(task_filter or "all").strip()
    if raw.lower() in {"", "all"}:
        return list(specs)

    requested = [x.strip() for x in raw.split(",") if x.strip()]
    if not requested:
        raise ValueError(
            f"Invalid downstream task filter for suite {suite!r}: {task_filter!r}. "
            "Use 'all' or a comma-separated list of task labels."
        )

    by_key = {_normalize_task_label(s.label): s for s in specs}
    missing = [name for name in requested if _normalize_task_label(name) not in by_key]
    if missing:
        available = ", ".join(s.label for s in specs)
        raise ValueError(
            f"Unknown downstream task(s) for suite {suite!r}: {missing}. "
            f"Available: {available}"
        )

    out: List[DownstreamTaskSpec] = []
    seen = set()
    for name in requested:
        key = _normalize_task_label(name)
        if key in seen:
            continue
        seen.add(key)
        out.append(by_key[key])
    return out


def _filter_specs_for_task_exclude_filter(
    specs: Sequence[DownstreamTaskSpec],
    task_exclude_filter: str,
) -> List[DownstreamTaskSpec]:
    raw = str(task_exclude_filter or "").strip()
    if raw.lower() in {"", "none"}:
        return list(specs)

    requested = {_normalize_task_label(x) for x in raw.split(",") if x.strip()}
    known = {_normalize_task_label(s.label) for s in _default_specs()}
    missing = sorted(requested - known)
    if missing:
        raise ValueError(
            f"Unknown downstream task(s) to exclude: {missing}. "
            f"Available: {', '.join(s.label for s in _default_specs())}"
        )
    return [s for s in specs if _normalize_task_label(s.label) not in requested]


def _validate_specs(specs: Sequence[DownstreamTaskSpec]) -> None:
    labels = [s.label for s in specs]
    if len(labels) != EXPECTED_DOWNSTREAM_TASK_COUNT:
        raise RuntimeError(
            f"Downstream registry is incomplete: expected {EXPECTED_DOWNSTREAM_TASK_COUNT} tasks, found {len(labels)}"
        )
    if len(set(labels)) != len(labels):
        raise RuntimeError(f"Downstream registry has duplicate labels: {labels}")
    allowed_metrics = {"mc", "em", "pass@10"}
    bad_metrics = sorted({s.metric for s in specs if s.metric not in allowed_metrics})
    if bad_metrics:
        raise RuntimeError(f"Downstream registry has unsupported metrics: {bad_metrics}")


def _validate_suite_specs(specs: Sequence[DownstreamTaskSpec], suite: str) -> None:
    if not specs:
        raise RuntimeError(f"Downstream suite {suite!r} resolved to zero tasks")
    labels = [s.label for s in specs]
    if len(set(labels)) != len(labels):
        raise RuntimeError(f"Downstream suite {suite!r} has duplicate labels: {labels}")
    allowed_metrics = {"mc", "em", "pass@10"}
    bad_metrics = sorted({s.metric for s in specs if s.metric not in allowed_metrics})
    if bad_metrics:
        raise RuntimeError(f"Downstream suite {suite!r} has unsupported metrics: {bad_metrics}")


def _require_hf_dataset_deps():
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Downstream eval requires the `datasets` package for runtime adapters. "
            "Install it with `pip install datasets`."
        ) from exc
    try:
        from huggingface_hub import hf_hub_download, list_repo_files  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Downstream eval requires `huggingface_hub` for runtime adapters. "
            "Install it with `pip install huggingface_hub`."
        ) from exc
    return load_dataset, hf_hub_download, list_repo_files


def _rank_world() -> tuple[int, int]:
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    return rank, world


def _distributed_mean(values: List[float], device) -> float:
    if not values:
        return float("nan")
    t = torch.tensor(values, dtype=torch.float32, device=device)
    local_sum = t.sum()
    local_count = torch.tensor([len(values)], dtype=torch.float32, device=device)
    if dist.is_initialized():
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
    denom = float(local_count.item())
    return float(local_sum.item() / max(denom, 1.0))


def _local_total_for_rank(total: int, rank: int, world: int) -> int:
    if rank >= total:
        return 0
    return ((total - 1 - rank) // max(world, 1)) + 1


def _maybe_log_rank0_progress(
    spec: DownstreamTaskSpec,
    *,
    processed_local: int,
    total_local: int,
    total_global: int,
    start_time_s: float,
    cfg: DownstreamEvalConfig,
) -> None:
    if not bool(cfg.verbose):
        return
    rank, _ = _rank_world()
    if rank != 0 or total_local <= 0:
        return
    log_every = max(int(cfg.log_every), 1)
    if processed_local != 1 and processed_local % log_every != 0 and processed_local != total_local:
        return
    elapsed = max(0.0, time.perf_counter() - start_time_s)
    pct = 100.0 * float(processed_local) / float(max(total_local, 1))
    print0(
        f"[downstream][{spec.label}] rank0_shard={processed_local}/{total_local} "
        f"({pct:.1f}%) global_total={total_global} elapsed={elapsed:.1f}s"
    )


def _strip_and_fold_ws(text: str) -> str:
    return " ".join(str(text if text is not None else "").strip().split())


def _normalize_for_em(text: str, qa_style: bool) -> str:
    base = _strip_and_fold_ws(text)
    if qa_style:
        return normalize_qa_answer_squad_style(base)
    return base


def _truncate_on_stops(text: str, stop_sequences: Optional[Sequence[str]]) -> str:
    if not stop_sequences:
        return text
    best = None
    for s in stop_sequences:
        if not s:
            continue
        i = text.find(s)
        if i >= 0 and (best is None or i < best):
            best = i
    if best is None:
        return text
    return text[:best]


def _em_match_exact(pred: str, answers: Sequence[str], qa_style: bool) -> bool:
    pred_n = _normalize_for_em(pred, qa_style)
    if pred_n == "":
        return False
    for ans in answers:
        ans_n = _normalize_for_em(ans, qa_style)
        if ans_n == "":
            continue
        if pred_n == ans_n:
            return True
    return False


def _em_match_prefix(pred: str, answers: Sequence[str], qa_style: bool) -> bool:
    pred_n = _normalize_for_em(pred, qa_style)
    if pred_n == "":
        return False
    for ans in answers:
        ans_n = _normalize_for_em(ans, qa_style)
        if ans_n == "":
            continue
        if pred_n == ans_n or pred_n.startswith(ans_n):
            return True
    return False


def _em_match(pred: str, answers: Sequence[str], qa_style: bool) -> bool:
    pred_n = _normalize_for_em(pred, qa_style)
    if pred_n == "":
        return False
    for ans in answers:
        ans_n = _normalize_for_em(ans, qa_style)
        if ans_n == "":
            continue
        if pred_n == ans_n:
            return True
        if ans_n in pred_n:
            return True
    return False


def _extract_first_line(text: str) -> str:
    lines = str(text or "").strip().splitlines()
    for line in lines:
        s = str(line).strip()
        if s != "":
            return s
    return ""


def _extract_lambada_first_word(text: str) -> str:
    line = _extract_first_line(text)
    if line == "":
        return ""
    word = line.split()[0]
    return word.strip(string.punctuation + "\"'`")


def _extract_first_number(text: str) -> Optional[str]:
    m = _FIRST_NUMBER_RE.search(str(text or ""))
    if m is None:
        return None
    return m.group(0).replace(",", "").strip()


def _score_em_prediction(
    *,
    task_label: str,
    pred: str,
    answers: Sequence[str],
    qa_style: bool,
) -> tuple[bool, str]:
    label = str(task_label or "").strip()
    pred_raw = _truncate_on_stops(str(pred or ""), None)
    parsed = _extract_first_line(pred_raw)
    if label == "LAMBADA":
        parsed = _extract_lambada_first_word(pred_raw)
        return bool(_em_match_exact(parsed, answers, qa_style=False)), parsed

    if label == "Arithmetic":
        num = _extract_first_number(parsed)
        if num is not None:
            parsed = num
            normalized_answers: List[str] = []
            for ans in answers:
                ans_num = _extract_first_number(str(ans))
                normalized_answers.append(ans_num if ans_num is not None else str(ans))
            return bool(_em_match_exact(parsed, normalized_answers, qa_style=False)), parsed
        return bool(_em_match_prefix(parsed, answers, qa_style=False)), parsed

    if label == "GSM8K":
        pn = _extract_gsm_numeric(parsed)
        if pn is None:
            pn = _extract_first_number(parsed)
        if pn is None:
            pn = _normalize_for_em(parsed, qa_style=False)
        ok = any(
            _normalize_for_em(str(a), qa_style=False) == _normalize_for_em(str(pn), qa_style=False)
            for a in answers
        )
        return bool(ok), str(pn)

    qa_flag = bool(qa_style) or (label in _QA_STYLE_TASKS)
    return bool(_em_match_prefix(parsed, answers, qa_style=qa_flag)), parsed


def _extract_gsm_numeric(text: str) -> Optional[str]:
    m = _GSM_RE.search(str(text))
    if m is None:
        return None
    return m.group(1).replace(",", "").strip()


def _extract_first_code_block_or_text(completion: str) -> str:
    matches = _CODE_BLOCK_RE.findall(str(completion or ""))
    if matches:
        return str(matches[0]).strip()
    return str(completion or "").strip()


def _pass_at_k(n: int, c: int, k: int) -> float:
    if n <= 0:
        return 0.0
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    # 1 - comb(n-c,k) / comb(n,k)
    num = math.comb(n - c, k)
    den = math.comb(n, k)
    return float(1.0 - (num / den))


def _sample_limit(data: List[dict], max_per_task: int, sample_seed: int = 1337) -> List[dict]:
    data = list(data)
    rng = random.Random(int(sample_seed))
    rng.shuffle(data)
    if max_per_task > 0:
        data = data[:max_per_task]
    return data


def _load_bundle_jsonl(eval_bundle_dir: str, dataset_uri: str) -> List[dict]:
    path = os.path.join(eval_bundle_dir, "eval_data", dataset_uri)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required downstream dataset is missing from eval bundle: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f]


def _load_arithmetic_split(filename: str) -> List[dict]:
    _, hf_hub_download, _ = _require_hf_dataset_deps()
    path = hf_hub_download(repo_id="EleutherAI/arithmetic", repo_type="dataset", filename=filename)
    out: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out.append(
                {
                    "context": str(row["context"]),
                    "continuation": str(row["completion"]),
                    "answers": [str(row["completion"])],
                }
            )
    if not out:
        raise RuntimeError(f"Arithmetic split is empty: {filename}")
    return out


def _load_cute_all_splits() -> List[dict]:
    load_dataset, hf_hub_download, list_repo_files = _require_hf_dataset_deps()
    files = list_repo_files(repo_id="leukas/cute", repo_type="dataset")
    data_files = [f for f in files if f.startswith("data/") and f.endswith(".parquet")]
    if not data_files:
        raise RuntimeError("CUTE dataset parquet files not found in leukas/cute")
    out: List[dict] = []
    for rel in sorted(data_files):
        path = hf_hub_download(repo_id="leukas/cute", repo_type="dataset", filename=rel)
        ds = load_dataset("parquet", data_files=path, split="train")
        for row in ds:
            out.append(
                {
                    "context": str(row["prompt"]),
                    "continuation": str(row["answer"]),
                    "answers": [str(row["answer"])],
                }
            )
    if not out:
        raise RuntimeError("CUTE dataset resolved to zero rows")
    return out


def _load_drop() -> List[dict]:
    load_dataset, hf_hub_download, _ = _require_hf_dataset_deps()
    path = hf_hub_download(
        repo_id="ucinlp/drop",
        repo_type="dataset",
        filename="data/validation-00000-of-00001.parquet",
    )
    ds = load_dataset("parquet", data_files=path, split="train")
    out: List[dict] = []
    for row in ds:
        spans = list((row.get("answers_spans") or {}).get("spans", []))
        answers = [str(s) for s in spans if str(s).strip()]
        if not answers:
            continue
        passage = str(row.get("passage", "")).strip()
        question = str(row.get("question", "")).strip()
        context = f"Passage: {passage}\n\nQuestion: {question}\nAnswer:"
        out.append({"context": context, "continuation": answers[0], "answers": answers})
    if not out:
        raise RuntimeError("DROP adapter produced zero rows")
    return out


def _flatten_hotpot_context(ctx: dict) -> str:
    titles = list((ctx or {}).get("title", []))
    sents = list((ctx or {}).get("sentences", []))
    chunks: List[str] = []
    for t, sent_list in zip(titles, sents):
        joined = " ".join([str(x) for x in sent_list])
        chunks.append(f"{t}: {joined}")
    return "\n".join(chunks)


def _load_hotpot() -> List[dict]:
    load_dataset, hf_hub_download, _ = _require_hf_dataset_deps()
    path = hf_hub_download(
        repo_id="hotpotqa/hotpot_qa",
        repo_type="dataset",
        filename="distractor/validation-00000-of-00001.parquet",
    )
    ds = load_dataset("parquet", data_files=path, split="train")
    out: List[dict] = []
    for row in ds:
        answer = str(row.get("answer", "")).strip()
        if answer == "":
            continue
        question = str(row.get("question", "")).strip()
        passage = _flatten_hotpot_context(row.get("context") or {})
        context = f"Passage: {passage}\n\nQuestion: {question}\nAnswer:"
        out.append({"context": context, "continuation": answer, "answers": [answer]})
    if not out:
        raise RuntimeError("HotpotQA adapter produced zero rows")
    return out


def _load_mbpp() -> List[dict]:
    load_dataset, hf_hub_download, _ = _require_hf_dataset_deps()
    path = hf_hub_download(
        repo_id="google-research-datasets/mbpp",
        repo_type="dataset",
        filename="sanitized/test-00000-of-00001.parquet",
    )
    ds = load_dataset("parquet", data_files=path, split="train")
    out: List[dict] = []
    for row in ds:
        out.append(
            {
                "prompt": str(row.get("prompt", "")),
                "test_imports": [str(x) for x in list(row.get("test_imports", []))],
                "test_list": [str(x) for x in list(row.get("test_list", []))],
            }
        )
    if not out:
        raise RuntimeError("MBPP adapter produced zero rows")
    return out


def _load_humaneval(eval_bundle_dir: str) -> List[dict]:
    rows = _load_bundle_jsonl(eval_bundle_dir, "programming/human_eval.jsonl")
    out: List[dict] = []
    for row in rows:
        out.append(
            {
                "prompt": str(row.get("prompt", "")),
                "entry_point": str(row.get("entry_point", "")),
                "test": str(row.get("test", "")),
            }
        )
    if not out:
        raise RuntimeError("HumanEval adapter produced zero rows")
    return out


def _convert_bundle_em_rows(spec: DownstreamTaskSpec, rows: List[dict]) -> List[dict]:
    out: List[dict] = []
    for row in rows:
        if "context" in row and "continuation" in row:
            answers = [str(row["continuation"])]
            if "aliases" in row and isinstance(row["aliases"], list):
                answers.extend([str(x) for x in row["aliases"] if str(x).strip()])
            if "answer" in row and str(row["answer"]).strip():
                answers.append(str(row["answer"]))
            out.append(
                {
                    "context": str(row["context"]),
                    "continuation": str(row["continuation"]),
                    "answers": answers,
                }
            )
            continue

        if "context" in row and "answer" in row:
            ans = str(row["answer"]).strip()
            aliases = [ans]
            if isinstance(row.get("aliases"), list):
                aliases.extend([str(x) for x in row["aliases"] if str(x).strip()])
            out.append({"context": str(row["context"]), "continuation": ans, "answers": aliases})
            continue

        if "query" in row and "choices" in row and "gold" in row:
            choices = list(row["choices"])
            gold = int(row["gold"])
            if gold < 0 or gold >= len(choices):
                raise ValueError(f"Invalid gold index in {spec.label}: gold={gold} choices={len(choices)}")
            answer = str(choices[gold])
            context = f"{str(row['query']).strip()}\nAnswer:"
            out.append({"context": context, "continuation": answer, "answers": [answer]})
            continue

        raise ValueError(
            f"Unsupported EM row schema for {spec.label}. Keys={sorted(row.keys())}"
        )
    return out


def _fewshot_indices(data_len: int, idx: int, k: int) -> List[int]:
    if k <= 0:
        return []
    if data_len <= 1:
        return []
    use_k = min(k, data_len - 1)
    rng = random.Random(1234 + idx)
    candidates = [i for i in range(data_len) if i != idx]
    return rng.sample(candidates, use_k)


def _build_em_prompt(data: List[dict], idx: int, num_fewshot: int, continuation_delimiter: str) -> tuple[str, List[str]]:
    item = data[idx]
    few_idxs = _fewshot_indices(len(data), idx, num_fewshot)
    delimiter = str(continuation_delimiter)
    prompt_delimiter = delimiter.rstrip()
    parts: List[str] = []
    for j in few_idxs:
        ex = data[j]
        parts.append(f"{str(ex['context']).strip()}{delimiter}{str(ex['continuation'])}")
    prefix = "\n\n".join(parts)
    if prefix:
        prefix += "\n\n"
    prompt = f"{prefix}{str(item['context']).strip()}{prompt_delimiter}"
    answers = list(item.get("answers", [item.get("continuation", "")]))
    answers = [str(a) for a in answers if str(a).strip()]
    return prompt, answers


def _score_em_task(
    spec: DownstreamTaskSpec,
    data: List[dict],
    device,
    cfg: DownstreamEvalConfig,
    generate_fn: GenerateFn,
    generate_batch_fn: Optional[GenerateBatchFn] = None,
    em_debug_hook: Optional[EMDebugHook] = None,
    compare_generate_fn: Optional[GenerateFn] = None,
    compare_model_label: str = "base",
) -> float:
    rank, world = _rank_world()
    local_scores: List[float] = []
    total_global = int(len(data))
    total_local = _local_total_for_rank(total_global, rank, world)
    processed_local = 0
    t0 = time.perf_counter()
    local_indices = list(range(rank, len(data), world))
    batch_size = max(int(cfg.em_batch_size), 1)
    use_batched = generate_batch_fn is not None and batch_size > 1
    pbar = None
    if rank == 0 and total_local > 0:
        pbar = tqdm(
            total=total_local,
            desc=f"downstream:{spec.label}",
            unit="ex",
            dynamic_ncols=True,
        )

    def _score_prediction(pred: str, answers: List[str]) -> tuple[bool, str]:
        return _score_em_prediction(
            task_label=str(spec.label),
            pred=str(pred),
            answers=answers,
            qa_style=bool(spec.qa_style_normalization),
        )

    if use_batched:
        for start in range(0, len(local_indices), batch_size):
            chunk = local_indices[start : start + batch_size]
            prompts: List[str] = []
            answers_list: List[List[str]] = []
            seeds: List[int] = []
            for idx in chunk:
                prompt, answers = _build_em_prompt(
                    data,
                    idx,
                    num_fewshot=int(spec.num_fewshot),
                    continuation_delimiter=str(spec.continuation_delimiter),
                )
                prompts.append(prompt)
                answers_list.append(list(answers))
                seeds.append(1234 + idx)

            generated_batch = generate_batch_fn(
                prompts,
                do_sample=bool(cfg.em_temperature > 0.0),
                max_new_tokens=int(cfg.em_max_new_tokens),
                temperature=float(cfg.em_temperature),
                top_p=float(cfg.em_top_p),
                top_k=int(cfg.em_top_k),
                num_return_sequences=1,
                stop_sequences=None,
                seeds=seeds,
                task_label=str(spec.label),
            )
            if len(generated_batch) != len(chunk):
                raise RuntimeError(
                    f"Batched generation returned wrong batch size for {spec.label}: "
                    f"expected={len(chunk)} got={len(generated_batch)}"
                )
            for i, _idx in enumerate(chunk):
                preds = generated_batch[i]
                pred = str(preds[0] if preds else "")
                ok, parsed_pred = _score_prediction(pred, answers_list[i])
                compare_pred = None
                compare_ok = None
                compare_parsed_pred = None
                if compare_generate_fn is not None and (not bool(ok)):
                    compare_generated = compare_generate_fn(
                        prompts[i],
                        do_sample=bool(cfg.em_temperature > 0.0),
                        max_new_tokens=int(cfg.em_max_new_tokens),
                        temperature=float(cfg.em_temperature),
                        top_p=float(cfg.em_top_p),
                        top_k=int(cfg.em_top_k),
                        num_return_sequences=1,
                        stop_sequences=None,
                        seed=1234 + int(_idx),
                        task_label=str(spec.label),
                    )
                    compare_pred = str(compare_generated[0] if compare_generated else "")
                    compare_ok, compare_parsed_pred = _score_prediction(compare_pred, answers_list[i])
                if em_debug_hook is not None:
                    em_debug_hook(
                        {
                            "task": str(spec.label),
                            "task_metric": str(spec.metric),
                            "task_source": str(spec.source_kind),
                            "example_idx": int(_idx),
                            "prompt": str(prompts[i]),
                            "answers": list(answers_list[i]),
                            "prediction": str(pred),
                            "parsed_prediction": str(parsed_pred),
                            "correct": bool(ok),
                            "qa_style_normalization": bool(spec.qa_style_normalization),
                            "compare_model_label": str(compare_model_label),
                            "compare_prediction": (None if compare_pred is None else str(compare_pred)),
                            "compare_parsed_prediction": (
                                None if compare_parsed_pred is None else str(compare_parsed_pred)
                            ),
                            "compare_correct": (None if compare_ok is None else bool(compare_ok)),
                        }
                    )
                local_scores.append(float(ok))
                processed_local += 1
                if pbar is not None:
                    pbar.update(1)
                _maybe_log_rank0_progress(
                    spec,
                    processed_local=processed_local,
                    total_local=total_local,
                    total_global=total_global,
                    start_time_s=t0,
                    cfg=cfg,
                )
    else:
        for idx in local_indices:
            prompt, answers = _build_em_prompt(
                data,
                idx,
                num_fewshot=int(spec.num_fewshot),
                continuation_delimiter=str(spec.continuation_delimiter),
            )
            answers = list(answers)
            generated = generate_fn(
                prompt,
                do_sample=bool(cfg.em_temperature > 0.0),
                max_new_tokens=int(cfg.em_max_new_tokens),
                temperature=float(cfg.em_temperature),
                top_p=float(cfg.em_top_p),
                top_k=int(cfg.em_top_k),
                num_return_sequences=1,
                stop_sequences=None,
                seed=1234 + idx,
                task_label=str(spec.label),
            )
            pred = str(generated[0] if generated else "")
            ok, parsed_pred = _score_prediction(pred, answers)
            compare_pred = None
            compare_ok = None
            compare_parsed_pred = None
            if compare_generate_fn is not None and (not bool(ok)):
                compare_generated = compare_generate_fn(
                    prompt,
                    do_sample=bool(cfg.em_temperature > 0.0),
                    max_new_tokens=int(cfg.em_max_new_tokens),
                    temperature=float(cfg.em_temperature),
                    top_p=float(cfg.em_top_p),
                    top_k=int(cfg.em_top_k),
                    num_return_sequences=1,
                    stop_sequences=None,
                    seed=1234 + int(idx),
                    task_label=str(spec.label),
                )
                compare_pred = str(compare_generated[0] if compare_generated else "")
                compare_ok, compare_parsed_pred = _score_prediction(compare_pred, answers)
            if em_debug_hook is not None:
                em_debug_hook(
                    {
                        "task": str(spec.label),
                        "task_metric": str(spec.metric),
                        "task_source": str(spec.source_kind),
                        "example_idx": int(idx),
                        "prompt": str(prompt),
                        "answers": list(answers),
                        "prediction": str(pred),
                        "parsed_prediction": str(parsed_pred),
                        "correct": bool(ok),
                        "qa_style_normalization": bool(spec.qa_style_normalization),
                        "compare_model_label": str(compare_model_label),
                        "compare_prediction": (None if compare_pred is None else str(compare_pred)),
                        "compare_parsed_prediction": (
                            None if compare_parsed_pred is None else str(compare_parsed_pred)
                        ),
                        "compare_correct": (None if compare_ok is None else bool(compare_ok)),
                    }
                )
            local_scores.append(float(ok))
            processed_local += 1
            if pbar is not None:
                pbar.update(1)
            _maybe_log_rank0_progress(
                spec,
                processed_local=processed_local,
                total_local=total_local,
                total_global=total_global,
                start_time_s=t0,
                cfg=cfg,
            )

    if pbar is not None:
        pbar.close()
    return _distributed_mean(local_scores, device)


def _score_humaneval_example(prompt: str, entry_point: str, test: str, completion: str) -> bool:
    code = _extract_first_code_block_or_text(completion)
    program = f"{prompt}{code}\n\n{test}\ncheck({entry_point})"
    result = execute_code(program)
    return bool(result.success)


def _score_mbpp_example(test_imports: List[str], test_list: List[str], completion: str) -> bool:
    code = _extract_first_code_block_or_text(completion)
    imports = "\n".join(test_imports)
    tests = "\n".join(test_list)
    program = f"{imports}\n\n{code}\n\n{tests}"
    result = execute_code(program)
    return bool(result.success)


def _score_passk_task(
    spec: DownstreamTaskSpec,
    data: List[dict],
    device,
    cfg: DownstreamEvalConfig,
    generate_fn: GenerateFn,
    generate_batch_fn: Optional[GenerateBatchFn] = None,
) -> float:
    rank, world = _rank_world()
    local_scores: List[float] = []
    stop_sequences = ["\nclass", "\ndef", "\n#", "\nif"]
    total_global = int(len(data))
    total_local = _local_total_for_rank(total_global, rank, world)
    processed_local = 0
    t0 = time.perf_counter()

    local_indices = list(range(rank, len(data), world))
    batch_size = max(int(cfg.coding_batch_size), 1)
    use_batched = generate_batch_fn is not None and batch_size > 1

    def _score_one(item: dict, prompt: str, completions: List[str]) -> float:
        outcomes: List[bool] = []
        for comp in completions:
            comp_cut = _truncate_on_stops(str(comp), stop_sequences)
            if spec.label == "HumanEval":
                ok = _score_humaneval_example(
                    prompt=prompt,
                    entry_point=str(item["entry_point"]),
                    test=str(item["test"]),
                    completion=comp_cut,
                )
            elif spec.label == "MBPP":
                ok = _score_mbpp_example(
                    test_imports=list(item["test_imports"]),
                    test_list=list(item["test_list"]),
                    completion=comp_cut,
                )
            else:
                raise ValueError(f"Unsupported pass@10 task: {spec.label}")
            outcomes.append(bool(ok))
        n = int(cfg.coding_num_samples)
        c = int(sum(1 for x in outcomes if x))
        return float(_pass_at_k(n=n, c=c, k=int(cfg.coding_pass_k)))

    if use_batched:
        for start in range(0, len(local_indices), batch_size):
            chunk = local_indices[start : start + batch_size]
            prompts = [str(data[idx]["prompt"]) for idx in chunk]
            seeds = [9871 + idx for idx in chunk]
            generated_batch = generate_batch_fn(
                prompts,
                do_sample=True,
                max_new_tokens=int(cfg.coding_max_new_tokens),
                temperature=float(cfg.coding_temperature),
                top_p=float(cfg.coding_top_p),
                top_k=int(cfg.coding_top_k),
                num_return_sequences=int(cfg.coding_num_samples),
                stop_sequences=stop_sequences,
                seeds=seeds,
                task_label=str(spec.label),
            )
            if len(generated_batch) != len(chunk):
                raise RuntimeError(
                    f"Batched generation returned wrong batch size for {spec.label}: "
                    f"expected={len(chunk)} got={len(generated_batch)}"
                )
            for i, idx in enumerate(chunk):
                item = data[idx]
                score = _score_one(item, prompts[i], generated_batch[i])
                local_scores.append(float(score))
                processed_local += 1
                _maybe_log_rank0_progress(
                    spec,
                    processed_local=processed_local,
                    total_local=total_local,
                    total_global=total_global,
                    start_time_s=t0,
                    cfg=cfg,
                )
    else:
        for idx in local_indices:
            item = data[idx]
            prompt = str(item["prompt"])
            completions = generate_fn(
                prompt,
                do_sample=True,
                max_new_tokens=int(cfg.coding_max_new_tokens),
                temperature=float(cfg.coding_temperature),
                top_p=float(cfg.coding_top_p),
                top_k=int(cfg.coding_top_k),
                num_return_sequences=int(cfg.coding_num_samples),
                stop_sequences=stop_sequences,
                seed=9871 + idx,
                task_label=str(spec.label),
            )
            score = _score_one(item, prompt, completions)
            local_scores.append(float(score))
            processed_local += 1
            _maybe_log_rank0_progress(
                spec,
                processed_local=processed_local,
                total_local=total_local,
                total_global=total_global,
                start_time_s=t0,
                cfg=cfg,
            )

    return _distributed_mean(local_scores, device)


def _load_spec_data(spec: DownstreamTaskSpec, eval_bundle_dir: str) -> List[dict]:
    if spec.source_kind == "bundle":
        rows = _load_bundle_jsonl(eval_bundle_dir, spec.bundle_dataset_uri)
        if spec.metric == "em":
            return _convert_bundle_em_rows(spec, rows)
        return rows

    if spec.source_kind == "cute":
        return _load_cute_all_splits()

    if spec.source_kind == "drop":
        return _load_drop()

    if spec.source_kind == "hotpot":
        return _load_hotpot()

    if spec.source_kind == "mbpp":
        return _load_mbpp()

    if spec.source_kind == "humaneval":
        return _load_humaneval(eval_bundle_dir)

    raise ValueError(f"Unsupported source kind for direct load: {spec.source_kind}")


def _score_arithmetic(
    spec: DownstreamTaskSpec,
    device,
    cfg: DownstreamEvalConfig,
    generate_fn: GenerateFn,
    generate_batch_fn: Optional[GenerateBatchFn] = None,
    em_debug_hook: Optional[EMDebugHook] = None,
    compare_generate_fn: Optional[GenerateFn] = None,
    compare_model_label: str = "base",
) -> tuple[float, Dict[str, float]]:
    split_map = {
        "2da": "data/two_digit_addition.jsonl",
        "2dm": "data/two_digit_multiplication.jsonl",
        "2ds": "data/two_digit_subtraction.jsonl",
    }
    split_scores: Dict[str, float] = {}
    rank, _ = _rank_world()
    for split_name, filename in split_map.items():
        split_data = _sample_limit(
            _load_arithmetic_split(filename), int(cfg.max_per_task), int(cfg.sample_seed)
        )
        if bool(cfg.verbose) and rank == 0:
            print0(f"[downstream] task=Arithmetic split={split_name} examples={len(split_data)}")
        score = _score_em_task(
            spec,
            split_data,
            device,
            cfg,
            generate_fn,
            generate_batch_fn=generate_batch_fn,
            em_debug_hook=em_debug_hook,
            compare_generate_fn=compare_generate_fn,
            compare_model_label=compare_model_label,
        )
        split_scores[split_name] = score
    mean_score = float(sum(split_scores.values()) / len(split_scores))
    return mean_score, split_scores


def evaluate_downstream(
    device,
    eval_bundle_dir: str,
    cfg: DownstreamEvalConfig,
    generate_fn: GenerateFn,
    mc_score_fn: MCScoreFn,
    generate_batch_fn: Optional[GenerateBatchFn] = None,
    generation_timing_fn: Optional[GenerationTimingFn] = None,
    em_debug_hook: Optional[EMDebugHook] = None,
    compare_generate_fn: Optional[GenerateFn] = None,
    compare_model_label: str = "base",
    task_result_callback: Optional[TaskResultCallback] = None,
) -> Dict[str, object]:
    if not os.path.isdir(eval_bundle_dir):
        raise FileNotFoundError(f"eval_bundle dir is missing: {eval_bundle_dir}")

    suite_key = _resolve_suite_alias(str(getattr(cfg, "suite", "downstream")))
    all_specs = _default_specs()
    _validate_specs(all_specs)
    specs = _select_specs_for_suite(all_specs, suite_key)
    _validate_suite_specs(specs, suite_key)
    specs = _filter_specs_for_task_filter(specs, str(getattr(cfg, "task_filter", "all")), suite_key)
    specs = _filter_specs_for_task_exclude_filter(
        specs,
        str(getattr(cfg, "task_exclude_filter", "")),
    )
    _validate_suite_specs(specs, f"{suite_key} (after task filter)")
    results: Dict[str, float] = {}
    categories: Dict[str, Dict[str, float]] = {}
    arithmetic_breakdown: Dict[str, float] = {}
    rank, _ = _rank_world()
    if bool(cfg.verbose) and rank == 0:
        print0(
            f"[downstream] starting suite with {len(specs)} tasks "
            f"(preset={str(getattr(cfg, 'preset', 'default'))!r}, "
            f"max_per_task={int(cfg.max_per_task)}, "
            f"task_filter={str(getattr(cfg, 'task_filter', 'all'))!r}, "
            f"task_exclude_filter={str(getattr(cfg, 'task_exclude_filter', ''))!r})"
        )

    for spec in specs:
        t_task = time.perf_counter()
        task_num_examples = None
        if bool(cfg.verbose) and rank == 0:
            print0(
                f"[downstream] start task={spec.label} metric={spec.metric} "
                f"fewshot={int(spec.num_fewshot)} source={spec.source_kind}"
            )
        if spec.source_kind == "arithmetic":
            score, split_scores = _score_arithmetic(
                spec,
                device,
                cfg,
                generate_fn,
                generate_batch_fn=generate_batch_fn,
                em_debug_hook=em_debug_hook,
                compare_generate_fn=compare_generate_fn,
                compare_model_label=compare_model_label,
            )
            results[spec.label] = score
            arithmetic_breakdown = split_scores
        elif spec.metric == "mc":
            data = _sample_limit(
                _load_spec_data(spec, eval_bundle_dir), int(cfg.max_per_task), int(cfg.sample_seed)
            )
            task_num_examples = len(data)
            if bool(cfg.verbose) and rank == 0:
                print0(f"[downstream] task={spec.label} examples={len(data)}")
            task_meta = {
                "label": spec.label,
                "task_type": spec.bundle_task_type,
                "num_fewshot": int(spec.num_fewshot),
                "continuation_delimiter": str(spec.continuation_delimiter),
            }
            score = float(mc_score_fn(data, task_meta))
            results[spec.label] = score
        elif spec.metric == "em":
            data = _sample_limit(
                _load_spec_data(spec, eval_bundle_dir), int(cfg.max_per_task), int(cfg.sample_seed)
            )
            task_num_examples = len(data)
            if bool(cfg.verbose) and rank == 0:
                print0(f"[downstream] task={spec.label} examples={len(data)}")
            score = _score_em_task(
                spec,
                data,
                device,
                cfg,
                generate_fn,
                generate_batch_fn=generate_batch_fn,
                em_debug_hook=em_debug_hook,
                compare_generate_fn=compare_generate_fn,
                compare_model_label=compare_model_label,
            )
            results[spec.label] = score
        elif spec.metric == "pass@10":
            data = _sample_limit(
                _load_spec_data(spec, eval_bundle_dir), int(cfg.max_per_task), int(cfg.sample_seed)
            )
            task_num_examples = len(data)
            if bool(cfg.verbose) and rank == 0:
                print0(f"[downstream] task={spec.label} examples={len(data)}")
            score = _score_passk_task(spec, data, device, cfg, generate_fn, generate_batch_fn=generate_batch_fn)
            results[spec.label] = score
        else:
            raise ValueError(f"Unsupported downstream metric: {spec.metric}")

        categories.setdefault(spec.category, {})[spec.label] = results[spec.label]
        if rank == 0:
            elapsed = max(0.0, time.perf_counter() - t_task)
            if task_result_callback is not None:
                task_result_callback({
                    "suite": suite_key,
                    "preset": str(getattr(cfg, "preset", "default")),
                    "task_exclude_filter": str(getattr(cfg, "task_exclude_filter", "")),
                    "task": str(spec.label),
                    "category": str(spec.category),
                    "metric": str(spec.metric),
                    "source_kind": str(spec.source_kind),
                    "score": float(results[spec.label]),
                    "num_examples": task_num_examples,
                    "elapsed_seconds": elapsed,
                    "max_per_task": int(cfg.max_per_task),
                    "sample_seed": int(cfg.sample_seed),
                    "arithmetic_splits": dict(arithmetic_breakdown) if spec.source_kind == "arithmetic" else None,
                })
            if bool(cfg.verbose):
                print0(
                    f"[downstream] done task={spec.label} "
                    f"score={float(results[spec.label]):.4f} elapsed={elapsed:.1f}s"
                )
                if generation_timing_fn is not None:
                    timing = generation_timing_fn(str(spec.label))
                    if timing is not None:
                        gen_s = float(timing.get("generate_s", 0.0))
                        dec_s = float(timing.get("decode_s", 0.0))
                        tot_s = max(gen_s + dec_s, 1e-9)
                        dec_pct = 100.0 * dec_s / tot_s
                        calls = int(timing.get("calls", 0.0))
                        comps = int(timing.get("completions", 0.0))
                        print0(
                            f"[downstream] timing task={spec.label} "
                            f"generate_s={gen_s:.2f} decode_s={dec_s:.2f} "
                            f"decode_pct={dec_pct:.1f}% calls={calls} completions={comps}"
                        )
            else:
                print0(
                    f"Downstream {spec.label}: {float(results[spec.label]):.4f} "
                    f"(metric={spec.metric}, {elapsed:.1f}s)"
                )

    downstream_metric = float(sum(results.values()) / max(len(results), 1))
    out: Dict[str, object] = {
        "results": results,
        "downstream_metric": downstream_metric,
        "categories": categories,
        "suite": suite_key,
        "preset": str(getattr(cfg, "preset", "default")),
        "max_per_task": int(cfg.max_per_task),
        "sample_seed": int(cfg.sample_seed),
        "task_filter": str(getattr(cfg, "task_filter", "all")),
        "task_exclude_filter": str(getattr(cfg, "task_exclude_filter", "")),
        "selected_task_labels": [s.label for s in specs],
    }
    if arithmetic_breakdown:
        out["arithmetic_splits"] = arithmetic_breakdown
    return out
