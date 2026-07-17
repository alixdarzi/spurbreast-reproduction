from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_TABLE2 = {
    "training": {"accuracy": 0.99, "ppv": 0.98, "npv": 0.99},
    "validation": {"accuracy": 0.99, "ppv": 1.00, "npv": 0.98},
    "test": {"accuracy": 0.52, "ppv": 0.62, "npv": 0.41},
}
PRIMARY_METRICS = ("accuracy", "ppv", "npv")
ALL_METRICS = (
    "accuracy",
    "ppv",
    "npv",
    "sensitivity",
    "specificity",
    "nll",
    "brier",
)
SPLITS = ("training", "validation", "test")
PALETTE = {"paper": "#3f4e5e", "reproduction": "#0072b2", "seed": "#56b4e9"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create de-identified aggregate tables and figures from locked evaluations."
    )
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "final_results"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[2025, 2026, 2027])
    parser.add_argument("--calibration-bins", type=int, default=10)
    return parser.parse_args()


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_float(value: object, context: str) -> float:
    if not _is_number(value) or not math.isfinite(float(value)):
        raise ValueError(f"Expected a finite number for {context}, got {value!r}")
    return float(value)


def find_evaluations(results_root: Path, seeds: Iterable[int]) -> dict[int, Path]:
    selected: dict[int, Path] = {}
    for seed in seeds:
        candidates = sorted(
            path
            for path in results_root.glob(
                f"locked_table2_field_strength_resnet50-seed{seed}-*/evaluation.json"
            )
            if (path.parent / "predictions.csv").is_file()
        )
        complete: list[Path] = []
        for path in candidates:
            data = json.loads(path.read_text(encoding="utf-8"))
            if all(split in data for split in SPLITS):
                complete.append(path)
        if len(complete) != 1:
            found = ", ".join(str(path) for path in complete) or "none"
            raise RuntimeError(
                f"Expected exactly one complete evaluation for seed {seed}; found {found}"
            )
        selected[seed] = complete[0]
    return selected


def load_evaluations(paths: dict[int, Path]) -> dict[int, dict]:
    evaluations = {
        seed: json.loads(path.read_text(encoding="utf-8")) for seed, path in paths.items()
    }
    for split in SPLITS:
        counts = {
            int(evaluations[seed][split]["slice_micro"]["n"]) for seed in evaluations
        }
        if len(counts) != 1:
            raise RuntimeError(f"Slice count differs across seeds for {split}: {sorted(counts)}")
    return evaluations


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def aggregate_evaluations(evaluations: dict[int, dict]) -> dict:
    output: dict[str, dict] = {"slice_micro": {}, "patient_macro": {}}
    for split in SPLITS:
        output["slice_micro"][split] = {}
        for metric in ALL_METRICS:
            values = [
                _finite_float(
                    evaluations[seed][split]["slice_micro"][metric],
                    f"seed {seed} {split} slice_micro {metric}",
                )
                for seed in sorted(evaluations)
            ]
            output["slice_micro"][split][metric] = summarize(values)
        output["patient_macro"][split] = {}
        for metric in ("accuracy", "ppv", "npv", "sensitivity", "specificity"):
            values = [
                _finite_float(
                    evaluations[seed][split]["patient_macro"][metric],
                    f"seed {seed} {split} patient_macro {metric}",
                )
                for seed in sorted(evaluations)
            ]
            output["patient_macro"][split][metric] = summarize(values)
    return output


def write_table2_csv(
    output_path: Path, evaluations: dict[int, dict], aggregate: dict
) -> None:
    fields = [
        "split",
        "metric",
        "paper",
        *[f"seed_{seed}" for seed in sorted(evaluations)],
        "reproduction_mean",
        "reproduction_sample_std",
        "difference_from_paper",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split in SPLITS:
            for metric in PRIMARY_METRICS:
                summary = aggregate["slice_micro"][split][metric]
                row: dict[str, object] = {
                    "split": split,
                    "metric": metric,
                    "paper": PAPER_TABLE2[split][metric],
                    "reproduction_mean": summary["mean"],
                    "reproduction_sample_std": summary["sample_std"],
                    "difference_from_paper": summary["mean"]
                    - PAPER_TABLE2[split][metric],
                }
                for seed in sorted(evaluations):
                    row[f"seed_{seed}"] = evaluations[seed][split]["slice_micro"][metric]
                writer.writerow(row)


def write_field_strength_csv(output_path: Path, evaluations: dict[int, dict]) -> list[dict]:
    rows: list[dict] = []
    for seed in sorted(evaluations):
        for split in SPLITS:
            strata = evaluations[seed][split]["field_strength_stratified"]
            for field_strength, metrics in sorted(strata.items()):
                row = {"seed": seed, "split": split, "field_strength": field_strength}
                for metric in (*ALL_METRICS, "predicted_positive_rate", "n"):
                    if metric in metrics:
                        row[metric] = metrics[metric]
                rows.append(row)
    fields = [
        "seed",
        "split",
        "field_strength",
        "n",
        *ALL_METRICS,
        "predicted_positive_rate",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def calibration_rows(
    prediction_paths: dict[int, Path], bins: int
) -> tuple[list[dict], list[dict]]:
    if bins < 2:
        raise ValueError("calibration-bins must be at least 2")
    rows: list[dict] = []
    summaries: list[dict] = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for seed, path in sorted(prediction_paths.items()):
        records = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
        for split in SPLITS:
            split_records = [row for row in records if row["split"] == split]
            strata = {"all": split_records}
            for value in sorted({row["field_strength_t"] for row in split_records}):
                strata[f"{float(value):g}T"] = [
                    row for row in split_records if row["field_strength_t"] == value
                ]
            for stratum, selected in strata.items():
                probabilities = np.asarray(
                    [float(row["positive_probability"]) for row in selected], dtype=float
                )
                truth = np.asarray([int(row["y_true"]) for row in selected], dtype=float)
                if probabilities.size == 0:
                    continue
                if not np.all((0.0 <= probabilities) & (probabilities <= 1.0)):
                    raise ValueError(f"Probabilities outside [0, 1] for seed {seed}")
                bin_index = np.minimum(np.digitize(probabilities, edges[1:-1]), bins - 1)
                weighted_gap = 0.0
                for index in range(bins):
                    mask = bin_index == index
                    if not np.any(mask):
                        continue
                    confidence = float(np.mean(probabilities[mask]))
                    event_rate = float(np.mean(truth[mask]))
                    count = int(np.sum(mask))
                    gap = abs(confidence - event_rate)
                    weighted_gap += count * gap
                    rows.append(
                        {
                            "seed": seed,
                            "split": split,
                            "field_strength": stratum,
                            "bin_index": index,
                            "bin_lower": edges[index],
                            "bin_upper": edges[index + 1],
                            "count": count,
                            "mean_predicted_probability": confidence,
                            "observed_positive_fraction": event_rate,
                            "absolute_gap": gap,
                        }
                    )
                summaries.append(
                    {
                        "seed": seed,
                        "split": split,
                        "field_strength": stratum,
                        "n": int(probabilities.size),
                        "ece": weighted_gap / probabilities.size,
                    }
                )
    return rows, summaries


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No rows available for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_table2(output_path: Path, evaluations: dict[int, dict], aggregate: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
    x = np.arange(len(SPLITS))
    offsets = np.linspace(-0.06, 0.06, len(evaluations))
    for axis, metric in zip(axes, PRIMARY_METRICS):
        paper = [PAPER_TABLE2[split][metric] for split in SPLITS]
        means = [aggregate["slice_micro"][split][metric]["mean"] for split in SPLITS]
        errors = [
            aggregate["slice_micro"][split][metric]["sample_std"] for split in SPLITS
        ]
        axis.plot(x, paper, "s--", color=PALETTE["paper"], label="Paper Table 2")
        axis.errorbar(
            x,
            means,
            yerr=errors,
            fmt="o-",
            capsize=4,
            color=PALETTE["reproduction"],
            label="Reproduction mean ± SD",
        )
        for offset, seed in zip(offsets, sorted(evaluations)):
            values = [evaluations[seed][split]["slice_micro"][metric] for split in SPLITS]
            axis.scatter(
                x + offset,
                values,
                s=24,
                facecolors="none",
                edgecolors=PALETTE["seed"],
                linewidths=1,
                label="Individual seeds" if seed == sorted(evaluations)[0] else None,
                zorder=3,
            )
        axis.set_title(metric.upper())
        axis.set_xticks(x, ["Train", "Validation", "Test"])
        axis.set_ylim(0.35, 1.03)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Slice-level metric")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Published Table 2 versus locked three-seed reproduction")
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_field_strength(output_path: Path, rows: list[dict]) -> None:
    test_rows = [row for row in rows if row["split"] == "test"]
    fields = sorted({row["field_strength"] for row in test_rows})
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharex=True)
    for axis, metric, label in (
        (axes[0], "accuracy", "Accuracy"),
        (axes[1], "predicted_positive_rate", "Predicted-positive rate"),
    ):
        for index, field in enumerate(fields):
            values = [float(row[metric]) for row in test_rows if row["field_strength"] == field]
            offsets = np.linspace(-0.06, 0.06, len(values))
            axis.scatter(
                np.full(len(values), index) + offsets,
                values,
                s=34,
                facecolors="none",
                edgecolors=PALETTE["seed"],
                linewidths=1.2,
            )
            mean = statistics.fmean(values)
            sd = statistics.stdev(values) if len(values) > 1 else 0.0
            axis.errorbar(
                index,
                mean,
                yerr=sd,
                fmt="o",
                capsize=5,
                color=PALETTE["reproduction"],
                markersize=7,
            )
        axis.set_xticks(range(len(fields)), fields)
        axis.set_ylim(-0.03, 1.03)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Test behavior stratified by magnetic field strength")
    fig.text(
        0.5,
        0.01,
        "Open circles: seeds; filled point and whisker: mean ± sample SD",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(output_path: Path, rows: list[dict]) -> None:
    strata = ("all", "1.5T", "3T")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1), sharex=True, sharey=True)
    seeds = sorted({int(row["seed"]) for row in rows})
    for axis, stratum in zip(axes, strata):
        axis.plot([0, 1], [0, 1], "--", color=PALETTE["paper"], linewidth=1)
        for seed in seeds:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["split"] == "test"
                    and row["field_strength"] == stratum
                    and int(row["seed"]) == seed
                ),
                key=lambda row: int(row["bin_index"]),
            )
            axis.plot(
                [float(row["mean_predicted_probability"]) for row in selected],
                [float(row["observed_positive_fraction"]) for row in selected],
                marker="o",
                markersize=3,
                linewidth=1,
                alpha=0.65,
                label=f"Seed {seed}",
            )
        axis.set_title("All test slices" if stratum == "all" else stratum)
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.02, 1.02)
        axis.set_xlabel("Mean predicted probability")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Observed tumor-slice fraction")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(seeds), frameon=False)
    fig.suptitle("Test reliability diagrams (10 equal-width bins)")
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    evaluation_paths = find_evaluations(args.results_root, args.seeds)
    evaluations = load_evaluations(evaluation_paths)
    aggregate = aggregate_evaluations(evaluations)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    write_table2_csv(output_dir / "table2_comparison.csv", evaluations, aggregate)
    field_rows = write_field_strength_csv(
        output_dir / "field_strength_metrics.csv", evaluations
    )
    prediction_paths = {
        seed: evaluation_path.parent / "predictions.csv"
        for seed, evaluation_path in evaluation_paths.items()
    }
    bins, calibration_summary = calibration_rows(
        prediction_paths, args.calibration_bins
    )
    write_rows(output_dir / "calibration_bins.csv", bins)
    write_rows(output_dir / "calibration_summary.csv", calibration_summary)

    public_summary = {
        "seeds": sorted(evaluations),
        "paper_table2": PAPER_TABLE2,
        "aggregate": aggregate,
        "source_run_names": {
            str(seed): path.parent.name for seed, path in evaluation_paths.items()
        },
        "privacy": "Only de-identified aggregate metrics are included; raw predictions remain private.",
    }
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps(public_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_table2(output_dir / "table2_comparison.png", evaluations, aggregate)
    plot_field_strength(output_dir / "field_strength_shortcut.png", field_rows)
    plot_calibration(output_dir / "test_calibration.png", bins)
    print(f"Wrote de-identified summary tables and figures to {output_dir}")


if __name__ == "__main__":
    main()
