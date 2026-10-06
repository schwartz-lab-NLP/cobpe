"""Verify native tokenization and a tiny CPU model without downloading data."""

import argparse
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    import nanochat_compositional_rust
    import rustbpe
    import tiktoken
    import torch

    from nanochat.compositional import CompositionalSpec
    from nanochat.compositional_rust import build_rust_backend
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS

    # An in-memory byte vocabulary keeps the check independent of model artifacts.
    encoding = tiktoken.Encoding(
        name='installation-check', pat_str=r'[\s\S]',
        mergeable_ranks={bytes([i]): i for i in range(256)},
        special_tokens={name: 256 + i for i, name in enumerate(SPECIAL_TOKENS)},
    )
    spec = CompositionalSpec.from_dict({
        'version': 1, 'num_modifier_groups': 1,
        'modifier_group_sizes': [2], 'group_names': ['space_prefix'],
        'group_value_names': {'space_prefix': ['', ' ']},
        'default_modifier': [0], 'entries': [],
    })
    with tempfile.TemporaryDirectory(prefix='cobpe-check-') as directory:
        RustBPETokenizer(encoding, '<|bos|>').save(directory)
        backend = build_rust_backend(spec, tokenizer_dir=directory)
        if backend is None:
            raise RuntimeError('The native CoBPE runtime did not load.')
        text = 'Hello, CoBPE! café 世界'
        ids, modifiers = backend.process_text(text)
        if backend.decode_with_modifiers(ids, modifiers) != text:
            raise RuntimeError('Native tokenizer roundtrip failed.')
        print(f'Native CoBPE runtime {nanochat_compositional_rust.__version__}: roundtrip passed')

    rustbpe.Tokenizer()
    print('BPE trainer: loaded')
    model = GPT(GPTConfig(
        sequence_len=64, vocab_size=encoding.n_vocab, n_layer=1,
        n_head=2, n_kv_head=2, n_embd=32, architecture_preset='vanilla',
        modifier_group_sizes=(2,),
    ))
    model.init_weights()
    tokens = torch.tensor([ids], dtype=torch.long)
    rows = torch.tensor([modifiers], dtype=torch.long)
    loss = model(
        tokens[:, :-1], targets=tokens[:, 1:], modifier_ids=rows[:, :-1],
        target_modifier_ids=rows[:, 1:], loss_impl='ce',
    )
    if not torch.isfinite(loss):
        raise RuntimeError('Tiny model produced a nonfinite loss.')
    loss.backward()
    print('CoBPE model: CPU forward/backward passed')
    print(f'PyTorch {torch.__version__}; CUDA available: {torch.cuda.is_available()}')
    print('Installation check passed. See docs/reproduce.md for data and training commands.')


if __name__ == '__main__':
    main()
