import sys

import pytest

from nanochat import downstream_eval as de


def test_downstream_registry_exact_30_tasks_and_metric_mapping():
    specs = de.get_downstream_specs()
    assert len(specs) == 30

    expected = {
        "ARC-Easy": "mc",
        "ARC-Challenge": "mc",
        "Jeopardy": "em",
        "MMLU": "mc",
        "OpenbookQA": "mc",
        "TriviaQA": "em",
        "WikidataQA": "em",
        "Arithmetic": "em",
        "GSM8K": "em",
        "LSAT-AR": "mc",
        "Operators": "em",
        "Repeat-Copy-Logic": "em",
        "HumanEval": "pass@10",
        "MBPP": "pass@10",
        "BoolQ": "mc",
        "CoQA": "em",
        "DROP": "em",
        "HotpotQA": "em",
        "SQuAD": "em",
        "CommonsenseQA": "mc",
        "COPA": "mc",
        "PIQA": "mc",
        "Winograd": "mc",
        "Winogrande": "mc",
        "HellaSwag": "mc",
        "LAMBADA": "em",
        "Language Identification": "em",
        "CS Algorithms": "em",
        "CUTE": "em",
        "Dyck-Languages": "em",
    }
    actual = {spec.label: spec.metric for spec in specs}
    assert actual == expected


def test_downstream_lm_suite_is_em_only_subset():
    specs = de.get_downstream_specs("downstream_lm")
    assert len(specs) > 0
    assert all(spec.metric == "em" for spec in specs)
    labels = {spec.label for spec in specs}
    assert "LAMBADA" in labels
    assert "MMLU" not in labels


def test_downstream_task_filter_selects_subset_in_requested_order():
    specs = list(de.get_downstream_specs("downstream"))
    selected = de._filter_specs_for_task_filter(specs, "mmlu,  triviaqa, MMLU", "downstream")
    assert [s.label for s in selected] == ["MMLU", "TriviaQA"]


def test_downstream_task_filter_rejects_unknown_task():
    specs = list(de.get_downstream_specs("downstream_lm"))
    with pytest.raises(ValueError, match="Unknown downstream task"):
        de._filter_specs_for_task_filter(specs, "LAMBADA,NotATask", "downstream_lm")


def test_em_cleaner_and_match_generic_and_qa_style():
    assert de._normalize_for_em("  a   b  ", qa_style=False) == "a b"
    assert de._em_match("Answer:   United States of America", ["United States"], qa_style=False)
    assert de._em_match("The United States.", ["united states"], qa_style=True)
    assert not de._em_match("Canada", ["United States"], qa_style=True)


def test_benchmark_scoring_lambada_is_first_word_exact():
    ok, parsed = de._score_em_prediction(
        task_label="LAMBADA",
        pred="newspapers and read it aloud",
        answers=["newspapers"],
        qa_style=False,
    )
    assert ok and parsed == "newspapers"

    ok2, parsed2 = de._score_em_prediction(
        task_label="LAMBADA",
        pred="papers and newspapers",
        answers=["newspapers"],
        qa_style=False,
    )
    assert not ok2
    assert parsed2 == "papers"


def test_benchmark_scoring_uses_prefix_match_not_contains():
    ok_prefix, _ = de._score_em_prediction(
        task_label="CUTE",
        pred="def then more words",
        answers=["def"],
        qa_style=False,
    )
    assert ok_prefix

    ok_not_contains, parsed = de._score_em_prediction(
        task_label="CUTE",
        pred="prefix def suffix",
        answers=["def"],
        qa_style=False,
    )
    assert not ok_not_contains
    assert parsed == "prefix def suffix"


def test_benchmark_scoring_qa_normalized_exact():
    ok, _ = de._score_em_prediction(
        task_label="SQuAD",
        pred="The United States. and more context",
        answers=["United States"],
        qa_style=True,
    )
    assert ok

    not_ok, _ = de._score_em_prediction(
        task_label="SQuAD",
        pred="In the United States",
        answers=["United States"],
        qa_style=True,
    )
    assert not not_ok


def test_benchmark_scoring_arithmetic_extracts_first_number():
    ok, parsed = de._score_em_prediction(
        task_label="Arithmetic",
        pred="42 and then more text",
        answers=["42"],
        qa_style=False,
    )
    assert ok
    assert parsed == "42"


def test_em_prompt_rstrips_continuation_delimiter():
    data = [{"context": "Question: 1+1?\nAnswer:", "continuation": "2", "answers": ["2"]}]
    prompt, answers = de._build_em_prompt(data, idx=0, num_fewshot=0, continuation_delimiter=" ")
    assert prompt.endswith("Answer:")
    assert answers == ["2"]


def test_arithmetic_aggregation_is_mean_of_three_splits(monkeypatch):
    def fake_load_arithmetic_split(filename):
        return [{"context": filename, "continuation": "x", "answers": ["x"]}]

    split_scores = {
        "two_digit_addition": 0.9,
        "two_digit_multiplication": 0.6,
        "two_digit_subtraction": 0.3,
    }

    def fake_score_em_task(spec, data, device, cfg, generate_fn, generate_batch_fn=None, **kwargs):
        context = data[0]["context"]
        for key, score in split_scores.items():
            if key in context:
                return score
        raise AssertionError(f"Unexpected context marker: {context}")

    monkeypatch.setattr(de, "_load_arithmetic_split", fake_load_arithmetic_split)
    monkeypatch.setattr(de, "_score_em_task", fake_score_em_task)

    spec = next(x for x in de.get_downstream_specs() if x.label == "Arithmetic")
    mean_score, details = de._score_arithmetic(spec, device="cpu", cfg=de.DownstreamEvalConfig(), generate_fn=lambda **_: [])
    assert details == {"2da": 0.9, "2dm": 0.6, "2ds": 0.3}
    assert mean_score == pytest.approx((0.9 + 0.6 + 0.3) / 3.0)


def test_pass_at_k_estimator():
    assert de._pass_at_k(n=20, c=0, k=10) == 0.0
    assert de._pass_at_k(n=20, c=20, k=10) == 1.0
    assert de._pass_at_k(n=10, c=1, k=1) == pytest.approx(0.1)


def test_missing_bundle_dataset_fails_loudly(tmp_path):
    eval_bundle_dir = tmp_path / "eval_bundle"
    (eval_bundle_dir / "eval_data").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="Required downstream dataset is missing"):
        de._load_bundle_jsonl(str(eval_bundle_dir), "does/not/exist.jsonl")


def test_missing_runtime_dataset_deps_are_actionable(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(RuntimeError, match="pip install datasets"):
        de._require_hf_dataset_deps()
