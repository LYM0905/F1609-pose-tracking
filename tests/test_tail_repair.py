from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from f1609.tail_repair import BODYPARTS, repair_tail


def prediction_frame() -> pd.DataFrame:
    columns = pd.MultiIndex.from_product([["scorer"], BODYPARTS, ["x", "y", "likelihood"]])
    rows = []
    for frame in range(3):
        values = []
        for channel, bodypart in enumerate(BODYPARTS):
            x = float(70 + frame * 10) if bodypart == "Tail" else float(channel * 10)
            values.extend([x, 0.0, 0.9])
        rows.append(values)
    return pd.DataFrame(rows, index=[0, 1, 2], columns=columns)


class TailRepairTests(unittest.TestCase):
    def test_stable_pose_is_unchanged(self) -> None:
        source = prediction_frame()
        repaired, log = repair_tail(source)
        pd.testing.assert_frame_equal(repaired, source)
        self.assertEqual(int(log["modified"].sum()), 0)

    def test_bad_tail_is_interpolated_and_only_tail_changes(self) -> None:
        source = prediction_frame()
        source.at[1, ("scorer", "Tail", "x")] = 400.0
        source.at[1, ("scorer", "Tail", "likelihood")] = 0.01
        repaired, log = repair_tail(source)
        self.assertAlmostEqual(float(repaired.at[1, ("scorer", "Tail", "x")]), 70.0)
        self.assertGreaterEqual(float(repaired.at[1, ("scorer", "Tail", "likelihood")]), 0.21)
        for bodypart in BODYPARTS[:-1]:
            for coord in ("x", "y", "likelihood"):
                pd.testing.assert_series_equal(repaired[("scorer", bodypart, coord)], source[("scorer", bodypart, coord)])
        self.assertEqual(log.loc[log["modified"], "frame"].tolist(), [1])

    def test_non_dlc_columns_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "MultiIndex"):
            repair_tail(pd.DataFrame({"Tail": [1.0]}))


if __name__ == "__main__":
    unittest.main()
