from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from f1609.checkpoint import verify_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_contract_detects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"f1609")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"checkpoint": {"filename": "model.pt", "sha256": hashlib.sha256(b"f1609").hexdigest().upper(), "bytes": 5}}), encoding="utf-8")
            self.assertTrue(verify_checkpoint(checkpoint, manifest)["passed"])
            checkpoint.write_bytes(b"changed")
            self.assertFalse(verify_checkpoint(checkpoint, manifest)["passed"])


if __name__ == "__main__":
    unittest.main()
