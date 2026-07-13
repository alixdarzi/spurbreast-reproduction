from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

import numpy as np


def _safe_divide(numerator: float, denominator: float) -> float:
    return float("nan") if denominator == 0 else numerator / denominator


def confusion_counts(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, int]:
    truth = np.asarray(y_true, dtype=np.int64)
    prediction = np.asarray(y_pred, dtype=np.int64)
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    if not np.isin(truth, [0, 1]).all() or not np.isin(prediction, [0, 1]).all():
        raise ValueError("Binary metrics require labels in {0, 1}")
    return {
        "tn": int(np.sum((truth == 0) & (prediction == 0))),
        "fp": int(np.sum((truth == 0) & (prediction == 1))),
        "fn": int(np.sum((truth == 1) & (prediction == 0))),
        "tp": int(np.sum((truth == 1) & (prediction == 1))),
    }


def binary_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    positive_probabilities: Sequence[float] | None = None,
) -> dict[str, float | int]:
    counts = confusion_counts(y_true, y_pred)
    tn, fp, fn, tp = (counts[name] for name in ("tn", "fp", "fn", "tp"))
    total = tn + fp + fn + tp
    result: dict[str, float | int] = {
        **counts,
        "n": total,
        "accuracy": _safe_divide(tp + tn, total),
        "ppv": _safe_divide(tp, tp + fp),
        "npv": _safe_divide(tn, tn + fn),
        "sensitivity": _safe_divide(tp, tp + fn),
        "specificity": _safe_divide(tn, tn + fp),
    }
    if positive_probabilities is not None:
        probabilities = np.asarray(positive_probabilities, dtype=np.float64)
        truth = np.asarray(y_true, dtype=np.float64)
        if probabilities.shape != truth.shape:
            raise ValueError("probabilities and y_true must have identical shapes")
        clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
        result["nll"] = float(
            -np.mean(truth * np.log(clipped) + (1 - truth) * np.log(1 - clipped))
        )
        result["brier"] = float(np.mean((probabilities - truth) ** 2))
    return result


def patient_macro_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    patient_ids: Sequence[str],
) -> dict[str, float]:
    if not (len(y_true) == len(y_pred) == len(patient_ids)):
        raise ValueError("All patient-macro inputs must have equal length")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, patient_id in enumerate(patient_ids):
        groups[patient_id].append(index)
    rows = []
    for indices in groups.values():
        rows.append(binary_metrics([y_true[i] for i in indices], [y_pred[i] for i in indices]))
    output: dict[str, float] = {"patients": float(len(rows))}
    for key in ("accuracy", "ppv", "npv", "sensitivity", "specificity"):
        values = np.asarray([row[key] for row in rows], dtype=float)
        output[key] = float(np.nanmean(values)) if not np.isnan(values).all() else float("nan")
    return output


def stratified_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probabilities: Sequence[float],
    strata: Sequence[float],
) -> dict[str, dict[str, float | int]]:
    if not (len(y_true) == len(y_pred) == len(probabilities) == len(strata)):
        raise ValueError("All stratified metric inputs must have equal length")
    output: dict[str, dict[str, float | int]] = {}
    for value in sorted(set(float(item) for item in strata if not math.isnan(float(item)))):
        indices = [index for index, item in enumerate(strata) if float(item) == value]
        output[f"{value:g}T"] = binary_metrics(
            [y_true[i] for i in indices],
            [y_pred[i] for i in indices],
            [probabilities[i] for i in indices],
        )
        output[f"{value:g}T"]["predicted_positive_rate"] = float(
            np.mean([y_pred[i] for i in indices])
        )
    return output


def patient_cluster_bootstrap(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    patient_ids: Sequence[str],
    resamples: int = 2000,
    seed: int = 2025,
) -> dict[str, dict[str, float]]:
    """Percentile intervals from patient-level cluster resampling."""
    if resamples <= 0:
        return {}
    if not (len(y_true) == len(y_pred) == len(patient_ids)):
        raise ValueError("All bootstrap inputs must have equal length")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, patient_id in enumerate(patient_ids):
        groups[patient_id].append(index)
    patients = sorted(groups)
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(resamples):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        indices = [index for patient in sampled for index in groups[str(patient)]]
        row = binary_metrics([y_true[i] for i in indices], [y_pred[i] for i in indices])
        for key in ("accuracy", "ppv", "npv", "sensitivity", "specificity"):
            if not math.isnan(float(row[key])):
                values[key].append(float(row[key]))
    return {
        key: {
            "lower": float(np.percentile(samples, 2.5)),
            "upper": float(np.percentile(samples, 97.5)),
        }
        for key, samples in values.items()
        if samples
    }
