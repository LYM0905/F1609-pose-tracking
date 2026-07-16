from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from f1609.tail_repair import TailRepairConfig, repair_tail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    args = parser.parse_args()
    prediction = pd.read_hdf(args.input)
    repaired, log = repair_tail(prediction, TailRepairConfig())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    repaired.to_hdf(args.output, key="df_with_missing", mode="w", format="fixed")
    repaired.to_csv(args.output.with_suffix(".csv"), encoding="utf-8-sig")
    log.to_csv(args.log, index=False, encoding="utf-8-sig")
    print(f"frames={len(repaired)} modified_tail_frames={int(log['modified'].sum())}")
    print(f"output={args.output}")
    print(f"log={args.log}")


if __name__ == "__main__":
    main()
