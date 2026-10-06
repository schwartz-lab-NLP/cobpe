import json
import os

import pytest
import torch

from cobpe.integrations.nanochat import entrypoints
from cobpe.integrations.nanochat.runtime import configure_output_base_dir
from cobpe.modeling.losses import CoBPELossConfig, compositional_lm_loss, validate_modifier_ids
from cobpe.tokenization.encoding import EncodedSequence, TokenCodec
from cobpe.tokenization.fingerprints import fingerprint_named_files
from nanochat.gpt import GPTConfig


class ToyCompositionalTokenizer:
    def has_compositional_mode(self):
        return True

    def get_default_modifier(self):
        return [0, 0]

    def get_num_modifier_groups(self):
        return 2

    def encode_with_modifiers(self, docs, prepend=None, num_threads=1, **_kwargs):
        def encode_one(text):
            ids = [ord(ch) % 17 + 3 for ch in text]
            modifiers = [[1 if ch.isupper() else 0, 1 if ch in ".!" else 0] for ch in text]
            if prepend is not None:
                ids = [int(prepend)] + ids
                modifiers = [self.get_default_modifier()] + modifiers
            return ids, modifiers

        if isinstance(docs, list):
            return [encode_one(doc) for doc in docs]
        return encode_one(docs)


def test_token_codec_lives_in_framework_neutral_package():
    seq = EncodedSequence([1, 2], [[0, 0], [1, 0]])
    codec = TokenCodec(ToyCompositionalTokenizer())
    assert isinstance(codec.normalize(seq), EncodedSequence)


def test_nanochat_is_framework_integration_peer(monkeypatch, tmp_path):
    assert entrypoints.pretraining == "scripts.base_train"
    assert entrypoints.evaluation == "scripts.base_eval"
    monkeypatch.setenv("NANOCHAT_BASE_DIR", "original")
    resolved = configure_output_base_dir(str(tmp_path / "runs"))
    assert resolved == str(tmp_path / "runs")
    assert os.environ["NANOCHAT_BASE_DIR"] == resolved
    assert configure_output_base_dir("") is None
    assert os.environ["NANOCHAT_BASE_DIR"] == resolved






def test_compositional_lm_loss_sums_group_relative_modifier_losses():
    torch.manual_seed(1)
    base_logits = torch.randn(2, 3, 7, requires_grad=True)
    base_labels = torch.randint(0, 7, (2, 3))
    group_logits = [
        torch.randn(2, 3, 2, requires_grad=True),
        torch.randn(2, 3, 4, requires_grad=True),
    ]
    modifier_labels = torch.stack(
        [
            torch.randint(0, 2, (2, 3)),
            torch.randint(0, 4, (2, 3)),
        ],
        dim=-1,
    )
    loss, parts = compositional_lm_loss(
        base_logits=base_logits,
        base_labels=base_labels,
        group_logits=group_logits,
        modifier_labels=modifier_labels,
        loss_mask=torch.ones(2, 3),
        config=CoBPELossConfig(),
    )
    expected_modifier_loss = sum(
        torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            modifier_labels[..., group_idx].reshape(-1),
        )
        for group_idx, logits in enumerate(group_logits)
    )
    assert torch.allclose(parts["modifier_loss"], expected_modifier_loss.detach())
    loss.backward()
    assert torch.isfinite(loss)
    assert set(parts) == {"base_loss", "modifier_loss"}
    assert base_logits.grad is not None
    assert group_logits[0].grad is not None


def test_validate_modifier_ids_rejects_out_of_group_range():
    labels = torch.tensor([[[0, 3]]])
    with pytest.raises(ValueError, match="group 1"):
        validate_modifier_ids(labels, [2, 3])








def test_canonical_modifier_conditioning_default_is_concat_gated_refine():
    config = GPTConfig()
    assert config.modifier_conditioning_mode == "concat_gated_refine"

    assert config.modifier_gate_mode == "per_group"


def test_local_tokenizer_fingerprint_changes_with_tokenizer_contents(tmp_path):
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    tokenizer_json.write_text('{"version": 1}', encoding="utf-8")
    first = fingerprint_named_files(str(tokenizer_dir), ("tokenizer.json",))["sha256"]
    tokenizer_json.write_text('{"version": 2}', encoding="utf-8")
    second = fingerprint_named_files(str(tokenizer_dir), ("tokenizer.json",))["sha256"]
    assert first != second
