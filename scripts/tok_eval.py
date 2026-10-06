"""Measure token count, byte compression, and roundtrip fidelity on UTF-8 files."""

import argparse
import json
import os
from pathlib import Path

from cobpe.tokenization.encoding import TokenCodec
from nanochat.tokenizer import get_tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer-dir', required=True)
    parser.add_argument('--text-file', type=Path, action='append', required=True, help='UTF-8 input; repeat for multiple documents')
    args = parser.parse_args()
    os.environ['NANOCHAT_TOKENIZER_DIR'] = os.path.abspath(args.tokenizer_dir)
    tokenizer = get_tokenizer()
    codec = TokenCodec(tokenizer)
    metrics = []
    for path in args.text_file:
        text = path.read_text(encoding='utf-8')
        encoded = codec.encode_text(text)
        num_bytes = len(text.encode('utf-8'))
        metrics.append({
            'file': str(path),
            'bytes': num_bytes,
            'tokens': len(encoded),
            'bytes_per_token': num_bytes / len(encoded) if len(encoded) else None,
            'roundtrip_ok': codec.decode(encoded) == text,
        })
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if not all(item['roundtrip_ok'] for item in metrics):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
