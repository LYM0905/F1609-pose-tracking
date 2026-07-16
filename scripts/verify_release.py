from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".mp4", ".avi", ".mov", ".h5"}
FORBIDDEN_TEXT = (
    "C:" + "\\" + "codex_dlc",
    "D:" + "\\" + "codex_dlc",
    "E:" + "\\" + "waseda",
    "175.155" + ".64.171",
)


def main() -> None:
    failures = []
    files = [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts and path.suffix != ".pyc"]
    for path in files:
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"forbidden artifact: {relative}")
        if path.stat().st_size > 5 * 1024 * 1024:
            failures.append(f"file exceeds 5 MiB: {relative}")
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"))
            except Exception as error:
                failures.append(f"python parse failed {relative}: {error}")
        if path.suffix.lower() in {".py", ".md", ".toml", ".yaml", ".yml", ".json"}:
            text = path.read_text(encoding="utf-8")
            for forbidden in FORBIDDEN_TEXT:
                if forbidden in text:
                    failures.append(f"private path or host in {relative}: {forbidden}")
    manifest = json.loads((ROOT / "artifacts" / "manifest.json").read_text(encoding="utf-8"))
    if manifest["checkpoint"]["sha256"] != "504521C281BF9C1E524B470C590300CFD1FB18D5535C7A119A53DD9DF8A4BC79":
        failures.append("checkpoint hash differs")
    result = {"passed": not failures, "file_count": len(files), "failures": failures}
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
