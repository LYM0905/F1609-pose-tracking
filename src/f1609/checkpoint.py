from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_checkpoint(path: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    checkpoint = Path(path)
    expected = load_manifest(manifest_path)["checkpoint"]
    actual = {"filename": checkpoint.name, "sha256": sha256(checkpoint), "bytes": checkpoint.stat().st_size}
    checks = {
        "filename": actual["filename"] == expected["filename"],
        "sha256": actual["sha256"] == expected["sha256"],
        "bytes": actual["bytes"] == int(expected["bytes"]),
    }
    return {"passed": all(checks.values()), "checks": checks, "actual": actual, "expected": expected}
