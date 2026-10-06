"""Stable content fingerprints for tokenizer artifacts and cache inputs."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence


def fingerprint_named_files(
    root: str,
    names: Sequence[str],
    *,
    required: Sequence[str] = (),
) -> dict[str, object]:
    """Hash selected files under ``root`` in a stable, filename-aware order."""

    root = os.path.abspath(os.path.expanduser(root))
    file_hashes: dict[str, str] = {}
    combined = hashlib.sha256()
    for name in names:
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            continue
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        file_digest = digest.hexdigest()
        file_hashes[name] = file_digest
        combined.update(name.encode("utf-8"))
        combined.update(b"\0")
        combined.update(file_digest.encode("ascii"))
        combined.update(b"\0")
    missing = [name for name in required if name not in file_hashes]
    if missing:
        raise FileNotFoundError(f"Missing required tokenizer artifacts in {root}: {missing}")
    return {
        "version": 1,
        "sha256": combined.hexdigest(),
        "files": file_hashes,
    }
