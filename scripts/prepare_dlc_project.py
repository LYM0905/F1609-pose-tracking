from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from f1609.checkpoint import verify_checkpoint

MODEL_SUBDIR = Path("dlc-models-pytorch/iteration-0/vpApr26-trainset95shuffle1/train")


def render(path: Path, replacements: dict[str, str]) -> str:
    text = path.read_text(encoding="utf-8")
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    verified = verify_checkpoint(args.checkpoint, ROOT / "artifacts" / "manifest.json")
    if not verified["passed"]:
        raise RuntimeError(f"Checkpoint verification failed: {verified['checks']}")
    if args.destination.exists():
        raise FileExistsError(args.destination)
    model_dir = args.destination / MODEL_SUBDIR
    test_dir = model_dir.parent / "test"
    model_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    replacements = {
        "__PROJECT_PATH__": str(args.destination.resolve()),
        "__POSE_CONFIG_PATH__": str((model_dir / "pytorch_config.yaml").resolve()),
    }
    (args.destination / "config.yaml").write_text(render(ROOT / "config" / "config.template.yaml", replacements), encoding="utf-8")
    (model_dir / "pytorch_config.yaml").write_text(render(ROOT / "config" / "pytorch_config.template.yaml", replacements), encoding="utf-8")
    (test_dir / "pose_cfg.yaml").write_text(render(ROOT / "config" / "pose_cfg.template.yaml", replacements), encoding="utf-8")
    shutil.copy2(args.checkpoint, model_dir / args.checkpoint.name)
    print(f"project={args.destination}")
    print(f"checkpoint={model_dir / args.checkpoint.name}")


if __name__ == "__main__":
    main()
