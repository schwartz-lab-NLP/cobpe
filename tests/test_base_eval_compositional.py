from types import SimpleNamespace

import torch

from cobpe.tokenization.encoding import EncodedSequence
from scripts import base_eval


class _ToyCompositionalTokenizer:
    def has_compositional_mode(self):
        return True

    def get_default_modifier(self):
        return [0]

    def encode_with_modifiers(self, text, prepend=None, append=None, **_kwargs):
        ids = [1]
        modifiers = [[3]]
        if prepend is not None:
            ids.insert(0, self.encode_special(prepend) if isinstance(prepend, str) else prepend)
            modifiers.insert(0, [0])
        return ids, modifiers

    def encode_special(self, token):
        return {"<|bos|>": 0, "<|assistant_end|>": 9}[token]

    def get_bos_token_id(self):
        return 0

    def decode_with_modifiers(self, ids, modifiers):
        return f"{ids}:{modifiers}"

    def decode(self, ids):
        return str(ids)


def test_downstream_prompt_encoding_uses_structured_compositional_api(monkeypatch):
    tokenizer = _ToyCompositionalTokenizer()
    model = SimpleNamespace(config=SimpleNamespace(sequence_len=8))
    captured = {}

    def fake_generate(_model, _tokenizer, rows, **_kwargs):
        captured["rows"] = rows
        return [["ok"]]

    monkeypatch.setattr(base_eval, "_generate_same_length_base_batch", fake_generate)
    _generate, generate_batch, _timing = base_eval._build_downstream_generate_fns(
        model, tokenizer, is_hf_model=False
    )
    result = generate_batch(
        ["question"],
        do_sample=False,
        max_new_tokens=1,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        num_return_sequences=1,
    )

    assert result == [["ok"]]
    assert captured["rows"] == [EncodedSequence([0, 1], [[0], [3]])]


def test_compositional_downstream_generation_feeds_and_decodes_modifiers():
    tokenizer = _ToyCompositionalTokenizer()

    class ToyModel:
        config = SimpleNamespace(
            sequence_len=4,
            n_kv_head=1,
            n_embd=2,
            n_head=1,
            n_layer=1,
        )

        def get_device(self):
            return torch.device("cpu")

        def forward(self, ids, *, kv_cache, modifier_ids, return_hidden):
            assert modifier_ids is not None
            kv_cache.advance(ids.size(1))
            logits = torch.zeros((*ids.shape, 10))
            logits[..., 2] = 1.0
            hidden = torch.zeros((*ids.shape, 2))
            return logits, hidden

        def get_modifier_logits(self, hidden, token_ids):
            logits = torch.zeros((hidden.size(0), 1, 4))
            logits[..., 3] = 1.0
            return [logits]

    result = base_eval._generate_same_length_base_batch(
        ToyModel(),
        tokenizer,
        [EncodedSequence([0, 1], [[0], [3]])],
        do_sample=False,
        max_new_tokens=1,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        num_return_sequences=1,
    )

    assert result == [["[2]:[[3]]"]]
