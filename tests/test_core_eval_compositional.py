from types import SimpleNamespace

import torch

from nanochat.core_eval import _modifier_predictions_match_with_suffix_boundary_rule


class _Tokenizer:
    spec = SimpleNamespace(group_to_idx={"suffix_punctuation": 1})

    def get_default_modifier(self):
        return [0, 0]


def test_core_lm_ignores_ungenerated_default_suffix_modifier():
    predicted = torch.tensor([[0, 1], [0, 1]])
    actual = torch.tensor([[0, 1], [0, 0]])
    assert _modifier_predictions_match_with_suffix_boundary_rule(predicted, actual, _Tokenizer())


def test_core_lm_still_requires_nondefault_suffix_modifier_match():
    predicted = torch.tensor([[0, 1], [0, 0]])
    actual = torch.tensor([[0, 1], [0, 1]])
    assert not _modifier_predictions_match_with_suffix_boundary_rule(predicted, actual, _Tokenizer())
