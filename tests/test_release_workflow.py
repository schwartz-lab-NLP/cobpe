"""Exercise published checkpoint loading and generation with real tokenizers."""

import json
from dataclasses import asdict

import pytest
import tiktoken
import torch

from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.engine import Engine
from nanochat.generation import decode_generated_batch, encode_prompt
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import SPECIAL_TOKENS, RustBPETokenizer, get_tokenizer_fingerprint


@pytest.mark.parametrize('compositional', [False, True], ids=['bpe', 'cobpe'])
def test_checkpoint_to_generation(tmp_path, monkeypatch, compositional):
    if compositional:
        pytest.importorskip('nanochat_compositional_rust')
    tokenizer_dir = tmp_path / 'tokenizer'
    encoding = tiktoken.Encoding(
        name='release-test', pat_str=r'[\s\S]',
        mergeable_ranks={bytes([i]): i for i in range(256)},
        special_tokens={name: 256 + i for i, name in enumerate(SPECIAL_TOKENS)},
    )
    base_tokenizer = RustBPETokenizer(encoding, '<|bos|>')
    base_tokenizer.save(str(tokenizer_dir))
    if compositional:
        (tokenizer_dir / 'compositional.json').write_text(json.dumps({
            'version': 1,
            'num_modifier_groups': 1,
            'modifier_group_sizes': [2],
            'group_names': ['space_prefix'],
            'group_value_names': {'space_prefix': ['', ' ']},
            'default_modifier': [0],
            'entries': [],
        }), encoding='utf-8')
    monkeypatch.setenv('NANOCHAT_TOKENIZER_DIR', str(tokenizer_dir))
    monkeypatch.setenv('NANOCHAT_BASE_DIR', str(tmp_path))
    config = GPTConfig(
        sequence_len=32, vocab_size=encoding.n_vocab, n_layer=1,
        n_head=2, n_kv_head=2, n_embd=32, architecture_preset='vanilla',
        modifier_group_sizes=(2,) if compositional else (),
    )
    model = GPT(config)
    model.init_weights()
    save_checkpoint(
        str(tmp_path / 'base_checkpoints' / 'tiny'), 1,
        model.state_dict(), None,
        {'model_config': asdict(config), 'tokenizer_fingerprint': get_tokenizer_fingerprint()},
    )
    loaded_model, tokenizer, _ = load_model('base', torch.device('cpu'), phase='eval', model_tag='tiny')
    prompt = encode_prompt(tokenizer, 'Hello')
    generated, masks = Engine(loaded_model, tokenizer).generate_batch(prompt, max_tokens=3, temperature=0)
    texts = decode_generated_batch(tokenizer, generated)
    assert texts[0].startswith('<|bos|>Hello')
    assert len(masks[0]) == len(prompt) + 3
    assert masks[0][-3:] == [1, 1, 1]
    if compositional:
        ids, modifiers = generated
        assert len(ids[0]) == len(modifiers[0])
        assert all(len(row) == 1 for row in modifiers[0])


def test_metadata_export_preserves_paper_surface_inventory(tmp_path, monkeypatch):
    """Freeze the original export's modifier IDs and composition map, not list order."""
    import hashlib
    import subprocess
    import sys

    tokenizer_dir = tmp_path / 'tokenizer'
    encoding = tiktoken.Encoding(
        name='smoke', pat_str=r'[\s\S]',
        mergeable_ranks={bytes([i]): i for i in range(256)},
        special_tokens={name: 256 + i for i, name in enumerate(SPECIAL_TOKENS)},
    )
    RustBPETokenizer(encoding, '<|bos|>').save(str(tokenizer_dir))
    monkeypatch.setenv('NANOCHAT_BASE_DIR', str(tmp_path / 'outputs'))
    proc = subprocess.run([
        sys.executable, '-m', 'cobpe', 'finalize-tokenizer',
        '--tokenizer-dir', str(tokenizer_dir),
        '--metadata-cache-dir', str(tmp_path / 'cache'),
        '--hf-tokenizer-dir', str(tmp_path / 'hf'),
    ], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    metadata = json.loads((tokenizer_dir / 'compositional.json').read_text())
    assert metadata['num_modifier_groups'] == 8
    assert sum(metadata['modifier_group_sizes']) == 124
    assert len(metadata['entries']) == 384
    canonical = {key: metadata[key] for key in [
        'default_indices', 'default_modifier', 'group_names', 'group_value_names',
        'inverse_entries', 'modifier_group_sizes', 'num_modifier_groups',
        'runtime_vocab_size', 'version',
    ]}
    canonical['entries'] = sorted(metadata['entries'], key=lambda item: json.dumps(item, sort_keys=True))
    digest = hashlib.sha256(json.dumps(
        canonical, sort_keys=True, ensure_ascii=False, separators=(',', ':'),
    ).encode()).hexdigest()
    # Reference export from the source checkout, before publication cleanup.
    assert digest == '8eafb974be9e4d8296efa7d2ff3923937915be9ddaeda850ba94b15a139f6d2e'
