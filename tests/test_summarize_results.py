from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_results.py"
SPEC = importlib.util.spec_from_file_location("summarize_results", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _metrics(value: float, n: int = 4) -> dict:
    return {
        "tn": 1,
        "fp": 1,
        "fn": 1,
        "tp": 1,
        "n": n,
        "accuracy": value,
        "ppv": value,
        "npv": value,
        "sensitivity": value,
        "specificity": value,
        "nll": 1.0 - value,
        "brier": (1.0 - value) / 2,
    }


def _evaluation(value: float) -> dict:
    output = {}
    for split in MODULE.SPLITS:
        output[split] = {
            "slice_micro": _metrics(value),
            "patient_macro": {
                "accuracy": value,
                "ppv": value,
                "npv": value,
                "sensitivity": value,
                "specificity": value,
            },
            "field_strength_stratified": {
                "1.5T": {**_metrics(value, n=2), "predicted_positive_rate": value},
                "3T": {**_metrics(value, n=2), "predicted_positive_rate": value},
            },
            "patient_cluster_95ci": {},
        }
    return output


def _write_result(root: Path, seed: int, value: float) -> Path:
    result_dir = root / f"locked_table2_field_strength_resnet50-seed{seed}-synthetic"
    result_dir.mkdir(parents=True)
    evaluation_path = result_dir / "evaluation.json"
    evaluation_path.write_text(json.dumps(_evaluation(value)), encoding="utf-8")
    with (result_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "split",
                "relative_path",
                "patient_id",
                "field_strength_t",
                "y_true",
                "y_pred",
                "positive_probability",
            ),
        )
        writer.writeheader()
        for split in MODULE.SPLITS:
            writer.writerows(
                [
                    {
                        "split": split,
                        "relative_path": "private/a.png",
                        "patient_id": "private-a",
                        "field_strength_t": 1.5,
                        "y_true": 0,
                        "y_pred": 0,
                        "positive_probability": 0.1,
                    },
                    {
                        "split": split,
                        "relative_path": "private/b.png",
                        "patient_id": "private-b",
                        "field_strength_t": 3.0,
                        "y_true": 1,
                        "y_pred": 1,
                        "positive_probability": 0.9,
                    },
                ]
            )
    return evaluation_path


def test_find_and_aggregate_evaluations(tmp_path: Path) -> None:
    for seed, value in ((2025, 0.8), (2026, 0.9), (2027, 1.0)):
        _write_result(tmp_path, seed, value)
    paths = MODULE.find_evaluations(tmp_path, [2025, 2026, 2027])
    aggregate = MODULE.aggregate_evaluations(MODULE.load_evaluations(paths))
    summary = aggregate["slice_micro"]["test"]["accuracy"]
    assert summary["mean"] == pytest.approx(0.9)
    assert summary["sample_std"] == pytest.approx(0.1)


def test_calibration_rows_are_deidentified(tmp_path: Path) -> None:
    evaluation_path = _write_result(tmp_path, 2025, 0.9)
    rows, summaries = MODULE.calibration_rows(
        {2025: evaluation_path.parent / "predictions.csv"}, bins=10
    )
    assert rows
    assert summaries
    assert summaries[0]["ece"] == pytest.approx(0.1)
    assert "patient_id" not in rows[0]
    assert "relative_path" not in rows[0]


def test_find_evaluations_rejects_ambiguous_seed(tmp_path: Path) -> None:
    _write_result(tmp_path, 2025, 0.9)
    second = tmp_path / "locked_table2_field_strength_resnet50-seed2025-second"
    second.mkdir()
    (second / "evaluation.json").write_text(json.dumps(_evaluation(0.8)), encoding="utf-8")
    (second / "predictions.csv").write_text("split\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="exactly one"):
        MODULE.find_evaluations(tmp_path, [2025])
