"""Command-line entry point for the paper's tokenizer and model workflows."""

import argparse
import runpy
import sys

COMMANDS = {
    'check-install': ('scripts.check_install', 'Check native tokenization and a tiny CPU model without downloading data'),
    'train-tokenizer': ('scripts.tok_train', 'Train BPE, normalized BPE, or SuperBPE'),
    'finalize-tokenizer': ('scripts.export_compositional_metadata', 'Build CoBPE metadata and finalize the vocabulary'),
    'inspect-tokenizer': ('scripts.tok_preview', 'Inspect tokenization and surface modifiers'),
    'evaluate-tokenizer': ('scripts.tok_eval', 'Measure compression and roundtrip fidelity'),
    'train': ('scripts.base_train', 'Pretrain a BPE or CoBPE language model'),
    'evaluate': ('scripts.base_eval', 'Evaluate a pretrained checkpoint'),
    'generate': ('scripts.generate', 'Continue a plain text prompt'),
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in {'-h', '--help'}:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('command', choices=COMMANDS)
        parser.epilog = '\n'.join(f'{name}: {description}' for name, (_, description) in COMMANDS.items())
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.print_help()
        return
    command = sys.argv[1]
    if command not in COMMANDS:
        raise SystemExit(f'Unknown command {command!r}. Run python -m cobpe --help for available commands.')
    sys.argv = [f'python -m cobpe {command}', *sys.argv[2:]]
    runpy.run_module(COMMANDS[command][0], run_name='__main__')


if __name__ == '__main__':
    main()
