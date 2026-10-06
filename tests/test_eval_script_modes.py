import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run_help(module_name: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        if "en_core_web_sm" in proc.stderr:
            pytest.skip("Skipping eval script help test: spaCy model en_core_web_sm is unavailable")
        assert proc.returncode == 0, proc.stderr
    return f"{proc.stdout}\n{proc.stderr}"


def test_base_eval_help_includes_downstream_mode():
    out = _run_help("scripts.base_eval")
    assert "downstream" in out


@pytest.mark.parametrize('command', [
    'check-install', 'train-tokenizer', 'finalize-tokenizer', 'inspect-tokenizer',
    'evaluate-tokenizer', 'train', 'evaluate', 'generate',
])
def test_public_command_help(command):
    proc = subprocess.run(
        [sys.executable, '-m', 'cobpe', command, '--help'],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert f'python -m cobpe {command}' in proc.stdout
