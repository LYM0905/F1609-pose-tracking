from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite
from typing import Any

import numpy as np
import pandas as pd


BODYPARTS = ("Head", "T1", "T2", "T3", "A3", "A5", "A7", "Tail")


@dataclass(frozen=True)
class TailRepairConfig:
    tail_likelihood_min: float = 0.20
    tail_jump_max_px: float = 90.0
    tail_from_a7_max_px: float = 120.0
    head_tail_collapse_px: float = 35.0
    neighbor_search_frames: int = 12
    bridge_search_frames: int = 48
    repaired_likelihood_min: float = 0.21


def _column(frame: pd.DataFrame, bodypart: str, coord: str) -> tuple[Any, ...]:
    matches = [column for column in frame.columns if isinstance(column, tuple) and len(column) >= 2 and column[-2] == bodypart and column[-1] == coord]
    if len(matches) != 1:
        raise ValueError(f"Expected one {bodypart}/{coord} column, found {len(matches)}")
    return matches[0]


def _validate(frame: pd.DataFrame) -> None:
    if not isinstance(frame.columns, pd.MultiIndex):
        raise ValueError("DLC prediction columns must be a MultiIndex")
    if frame.index.has_duplicates:
        raise ValueError("Prediction frame index must be unique")
    for bodypart in BODYPARTS:
        for coord in ("x", "y", "likelihood"):
            _column(frame, bodypart, coord)


def _point(frame: pd.DataFrame, index: int, bodypart: str) -> tuple[float, float, float]:
    return tuple(float(frame.at[index, _column(frame, bodypart, coord)]) for coord in ("x", "y", "likelihood"))


def _finite(point: tuple[float, float, float]) -> bool:
    return isfinite(point[0]) and isfinite(point[1])


def _distance(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return hypot(first[0] - second[0], first[1] - second[1]) if _finite(first) and _finite(second) else float("inf")


def tail_failure_reasons(frame: pd.DataFrame, index: int, config: TailRepairConfig) -> list[str]:
    tail, head, a7 = _point(frame, index, "Tail"), _point(frame, index, "Head"), _point(frame, index, "A7")
    reasons: list[str] = []
    if not _finite(tail):
        reasons.append("tail_nonfinite")
    if not isfinite(tail[2]) or tail[2] < config.tail_likelihood_min:
        reasons.append("tail_low_likelihood")
    if _distance(head, tail) < config.head_tail_collapse_px:
        reasons.append("head_tail_collapse")
    if _distance(a7, tail) > config.tail_from_a7_max_px:
        reasons.append("tail_far_from_A7")
    if index - 1 in frame.index and _distance(_point(frame, index - 1, "Tail"), tail) > config.tail_jump_max_px:
        reasons.append("tail_jump_from_prev")
    return reasons


def _stable(frame: pd.DataFrame, index: int, config: TailRepairConfig) -> bool:
    tail = _point(frame, index, "Tail")
    return not tail_failure_reasons(frame, index, config) and isfinite(tail[2]) and tail[2] >= config.tail_likelihood_min


def _anchor(frame: pd.DataFrame, index: int, direction: int, radius: int, config: TailRepairConfig) -> int | None:
    for offset in range(1, radius + 1):
        candidate = index + direction * offset
        if candidate in frame.index and _stable(frame, candidate, config):
            return candidate
    return None


def _candidate(frame: pd.DataFrame, index: int, radius: int, prefix: str, config: TailRepairConfig) -> tuple[tuple[float, float, float] | None, str]:
    before, after = _anchor(frame, index, -1, radius, config), _anchor(frame, index, 1, radius, config)
    if before is not None and after is not None:
        first, second = _point(frame, before, "Tail"), _point(frame, after, "Tail")
        ratio = (index - before) / (after - before)
        return (first[0] * (1 - ratio) + second[0] * ratio, first[1] * (1 - ratio) + second[1] * ratio, min(max(first[2], second[2], config.repaired_likelihood_min), 1.0)), f"{prefix}_interp_{before}_{after}"
    anchor = before if before is not None else after
    if anchor is not None:
        point = _point(frame, anchor, "Tail")
        side = "prev" if before is not None else "next"
        return (point[0], point[1], max(point[2], config.repaired_likelihood_min)), f"{prefix}_hold_{side}_{anchor}"
    return None, f"{prefix}_no_anchor"


def _set_tail(frame: pd.DataFrame, index: int, point: tuple[float, float, float], config: TailRepairConfig) -> None:
    values = {"x": point[0], "y": point[1], "likelihood": max(point[2], config.repaired_likelihood_min)}
    for coord, value in values.items():
        column = _column(frame, "Tail", coord)
        try:
            value = np.asarray(value).astype(frame[column].dtype).item()
        except (TypeError, ValueError):
            value = float(value)
        frame.at[index, column] = value


def repair_tail(predictions: pd.DataFrame, config: TailRepairConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    active = config or TailRepairConfig()
    _validate(predictions)
    repaired, logs = predictions.copy(deep=True), []
    for index in sorted(int(value) for value in repaired.index):
        reasons = tail_failure_reasons(repaired, index, active)
        if not reasons:
            logs.append({"frame": index, "modified": False, "reason": "", "method": ""})
            continue
        old = _point(repaired, index, "Tail")
        candidate, method = _candidate(repaired, index, active.neighbor_search_frames, "neighbor", active)
        if candidate is None:
            candidate, method = _candidate(repaired, index, active.bridge_search_frames, "bridge", active)
        row = {"frame": index, "modified": candidate is not None, "reason": ",".join(reasons), "method": method, "old_tail_x": old[0], "old_tail_y": old[1], "old_tail_likelihood": old[2]}
        if candidate is not None:
            _set_tail(repaired, index, candidate, active)
            new = _point(repaired, index, "Tail")
            row.update({"new_tail_x": new[0], "new_tail_y": new[1], "new_tail_likelihood": new[2], "move_px": _distance(old, new)})
        logs.append(row)
    return repaired, pd.DataFrame(logs)
