from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional


REVERSE_SEMANTIC_INDEX = (7, 6, 5, 4, 3, 2, 1, 0)


@dataclass(frozen=True)
class SemanticMarginResult:
    loss: torch.Tensor
    margins: torch.Tensor


@dataclass(frozen=True)
class MarginDetectorResult:
    threshold: float
    clean_count: int
    wrong_side_count: int
    clean_false_positive_rate: float
    wrong_side_recall: float
    roc_auc: float


@dataclass(frozen=True)
class GradientComparison:
    native_norm: float
    auxiliary_norm: float
    cosine: float
    equal_norm_weight: float


@dataclass(frozen=True)
class P0GateDecision:
    passed: bool
    checks: dict[str, bool]
    failed_checks: tuple[str, ...]


def evaluate_p0_gate(evidence: dict) -> P0GateDecision:
    contracts = evidence["contracts"]
    counts = evidence["counts"]
    detector = evidence["detector"]
    gradient = evidence["gradient"]
    safety = evidence["safety"]
    cosine = float(gradient["cosine"])
    native_norm = float(gradient["native_norm"])
    auxiliary_norm = float(gradient["auxiliary_norm"])
    equal_norm_weight = float(gradient["equal_norm_weight"])
    checks = {
        "all_contracts": bool(contracts) and all(bool(value) for value in contracts.values()),
        "row_count_1629": int(counts["rows"]) == 1629,
        "point_count_13032": int(counts["points"]) == 1629 * 8,
        "target114_exact": int(counts["target114"]) == 114,
        "wrong_side_support": int(counts["wrong_side"]) >= 8,
        "clean_support": int(counts["clean"]) >= 5000,
        "locked_rows_zero": int(counts["locked_rows"]) == 0,
        "wrong_side_auc": float(detector["roc_auc"]) >= 0.70,
        "wrong_side_recall_at_clean_fpr": float(detector["wrong_side_recall"]) >= 0.50,
        "clean_false_positive_rate": float(detector["clean_false_positive_rate"]) <= 0.075,
        "gradient_norms_finite_positive": math.isfinite(native_norm)
        and math.isfinite(auxiliary_norm)
        and native_norm > 0.0
        and auxiliary_norm > 0.0,
        "gradient_nonredundant": math.isfinite(cosine) and abs(cosine) <= 0.98,
        "gradient_weight_calibratable": math.isfinite(equal_norm_weight)
        and 1e-4 <= equal_norm_weight <= 1e4,
        "model_state_unchanged": bool(safety["model_state_unchanged"]),
        "optimizer_not_created": not bool(safety["optimizer_created"]),
        "checkpoint_not_created": not bool(safety["checkpoint_created"]),
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    return P0GateDecision(passed=not failed, checks=checks, failed_checks=failed)


def compare_loss_gradients(
    native_loss: torch.Tensor,
    auxiliary_loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
) -> GradientComparison:
    if not parameters:
        raise ValueError("At least one parameter is required")
    native_gradients = torch.autograd.grad(
        native_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    auxiliary_gradients = torch.autograd.grad(
        auxiliary_loss,
        parameters,
        retain_graph=False,
        allow_unused=True,
    )
    native_sq = torch.zeros((), dtype=torch.float64)
    auxiliary_sq = torch.zeros((), dtype=torch.float64)
    dot = torch.zeros((), dtype=torch.float64)
    for parameter, native, auxiliary in zip(
        parameters,
        native_gradients,
        auxiliary_gradients,
    ):
        native_value = torch.zeros_like(parameter) if native is None else native
        auxiliary_value = torch.zeros_like(parameter) if auxiliary is None else auxiliary
        native_flat = native_value.detach().double().reshape(-1).cpu()
        auxiliary_flat = auxiliary_value.detach().double().reshape(-1).cpu()
        native_sq += torch.dot(native_flat, native_flat)
        auxiliary_sq += torch.dot(auxiliary_flat, auxiliary_flat)
        dot += torch.dot(native_flat, auxiliary_flat)
    native_norm = float(torch.sqrt(native_sq).item())
    auxiliary_norm = float(torch.sqrt(auxiliary_sq).item())
    if native_norm == 0.0 or auxiliary_norm == 0.0:
        cosine = float("nan")
    else:
        cosine = float(dot.item() / (native_norm * auxiliary_norm))
    equal_norm_weight = (
        native_norm / auxiliary_norm if auxiliary_norm > 0.0 else float("nan")
    )
    return GradientComparison(
        native_norm=native_norm,
        auxiliary_norm=auxiliary_norm,
        cosine=cosine,
        equal_norm_weight=equal_norm_weight,
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def calibrate_low_margin_detector(
    margins: np.ndarray,
    *,
    clean_mask: np.ndarray,
    wrong_side_mask: np.ndarray,
    clean_false_positive_target: float = 0.05,
) -> MarginDetectorResult:
    values = np.asarray(margins, dtype=float)
    clean = np.asarray(clean_mask, dtype=bool)
    wrong_side = np.asarray(wrong_side_mask, dtype=bool)
    if values.ndim != 1 or clean.shape != values.shape or wrong_side.shape != values.shape:
        raise ValueError("Margins and masks must be one-dimensional arrays with equal shape")
    if not 0.0 < clean_false_positive_target < 1.0:
        raise ValueError("clean_false_positive_target must be between 0 and 1")
    if np.any(clean & wrong_side):
        raise ValueError("Clean and wrong-side masks overlap")
    if not np.isfinite(values).all():
        raise ValueError("Margins contain non-finite values")
    clean_values = values[clean]
    wrong_values = values[wrong_side]
    if len(clean_values) == 0 or len(wrong_values) == 0:
        raise ValueError("Both clean and wrong-side groups must be non-empty")

    threshold = float(np.quantile(clean_values, clean_false_positive_target))
    clean_false_positive_rate = float(np.mean(clean_values <= threshold))
    wrong_side_recall = float(np.mean(wrong_values <= threshold))

    scores = np.concatenate([-clean_values, -wrong_values])
    labels = np.concatenate(
        [np.zeros(len(clean_values), dtype=bool), np.ones(len(wrong_values), dtype=bool)]
    )
    ranks = _average_ranks(scores)
    positive_rank_sum = float(ranks[labels].sum())
    positive_count = len(wrong_values)
    negative_count = len(clean_values)
    roc_auc = (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)
    return MarginDetectorResult(
        threshold=threshold,
        clean_count=negative_count,
        wrong_side_count=positive_count,
        clean_false_positive_rate=clean_false_positive_rate,
        wrong_side_recall=wrong_side_recall,
        roc_auc=float(roc_auc),
    )


def semantic_margin_loss(
    heatmap: torch.Tensor,
    target_heatmap: torch.Tensor,
) -> SemanticMarginResult:
    if heatmap.shape != target_heatmap.shape:
        raise ValueError(
            f"Heatmap and target shape differ: {heatmap.shape} != {target_heatmap.shape}"
        )
    if heatmap.ndim != 4 or heatmap.shape[1] != 8:
        raise ValueError(f"Expected [batch, 8, height, width], got {heatmap.shape}")

    batch, channels, height, width = heatmap.shape
    flat_output = heatmap.reshape(batch, channels, height * width)
    flat_target = target_heatmap.reshape(batch, channels, height * width)
    own_positions = flat_target.argmax(dim=-1)
    own_scores = flat_output.gather(dim=-1, index=own_positions[..., None]).squeeze(-1)
    reverse_index = torch.tensor(
        REVERSE_SEMANTIC_INDEX,
        device=heatmap.device,
        dtype=torch.long,
    )
    reverse_positions = own_positions.index_select(dim=1, index=reverse_index)
    reverse_scores = flat_output.gather(
        dim=-1,
        index=reverse_positions[..., None],
    ).squeeze(-1)
    margins = own_scores - reverse_scores
    return SemanticMarginResult(loss=functional.softplus(-margins).mean(), margins=margins)
