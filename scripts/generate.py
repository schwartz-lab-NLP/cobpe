"""Generate raw text from a BPE or CoBPE pretraining checkpoint."""

import argparse
import os

import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init
from nanochat.engine import Engine
from nanochat.generation import decode_sequence, encode_prompt, generated_batch_to_sequences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt', required=True, help='Text to continue')
    parser.add_argument('--model-tag', required=True, help='Checkpoint subdirectory under base_checkpoints')
    parser.add_argument('--step', type=int, default=None, help='Checkpoint step (default: latest)')
    parser.add_argument('--tokenizer-dir', required=True, help='Exact tokenizer used for training')
    parser.add_argument('--output-base-dir', required=True, help='Directory containing base_checkpoints')
    parser.add_argument('--device-type', choices=['cpu', 'cuda', 'mps'], default=None)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.8, help='Zero selects greedy decoding')
    parser.add_argument('--top-k', type=int, default=None)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error('--max-new-tokens must be positive')
    if args.temperature < 0 or not 0 < args.top_p <= 1:
        parser.error('temperature must be nonnegative and top-p must be in (0, 1]')
    if args.top_k is not None and args.top_k < 1:
        parser.error('--top-k must be positive')
    os.environ['NANOCHAT_TOKENIZER_DIR'] = os.path.abspath(args.tokenizer_dir)
    os.environ['NANOCHAT_BASE_DIR'] = os.path.abspath(args.output_base_dir)
    ddp, rank, _, _, device = compute_init(args.device_type or autodetect_device_type())
    try:
        if ddp:
            parser.error('Run generation with python, without torchrun')
        model, tokenizer, _ = load_model('base', device, phase='eval', model_tag=args.model_tag, step=args.step)
        prompt = encode_prompt(tokenizer, args.prompt)
        if len(prompt) + args.max_new_tokens > model.config.sequence_len:
            parser.error('Prompt plus requested generation exceeds the checkpoint context length')
        dtype_context = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if device.type == 'cuda' else torch.no_grad()
        with dtype_context:
            batch, _ = Engine(model, tokenizer).generate_batch(
                prompt, max_tokens=args.max_new_tokens, temperature=args.temperature,
                top_k=args.top_k, top_p=args.top_p, seed=args.seed,
            )
        sequence = generated_batch_to_sequences(tokenizer, batch)[0]
        print(decode_sequence(tokenizer, sequence.slice(1)))
    finally:
        compute_cleanup()


if __name__ == '__main__':
    main()
