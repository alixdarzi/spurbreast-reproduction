from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spurbreast_repro.config import config_sha256, load_config, project_path  # noqa: E402
from spurbreast_repro.utils import git_commit, json_dumps, write_json  # noqa: E402


PRIMARY_IDS = ("H1", "H2", "H3", "H4")
FALLBACK_IDS = ("F1", "F2")
RUN_CONFIG_DIR = PROJECT_ROOT / "configs" / "sensitivity_runs"
REPORT_PATH = PROJECT_ROOT / "reports" / "tables" / "sensitivity_selection.json"


def run_config_path(identifier: str) -> Path:
    return RUN_CONFIG_DIR / f"{identifier}.yaml"


def completed_result(identifier: str) -> dict[str, Any] | None:
    config, config_path = load_config(run_config_path(identifier))
    expected_hash = config_sha256(config)
    name = config["experiment"]["name"]
    seed = int(config["experiment"]["seed"])
    result_root = project_path(config["paths"]["results"])
    for result_dir in sorted(
        result_root.glob(f"{name}-seed{seed}-*"), key=lambda path: path.name, reverse=True
    ):
        summary_path = result_dir / "summary.json"
        snapshot_path = result_dir / "config.yaml"
        if not summary_path.is_file() or not snapshot_path.is_file():
            continue
        snapshot, _ = load_config(snapshot_path)
        if config_sha256(snapshot) != expected_hash:
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "completed":
            continue
        return {
            "id": identifier,
            "run_id": result_dir.name,
            "config_path": config_path.relative_to(PROJECT_ROOT).as_posix(),
            "config_sha256": expected_hash,
            "best_epoch": int(summary["best_epoch"]),
            "best_val_accuracy": float(summary["best_val_accuracy"]),
            "best_val_nll": float(summary["best_val_nll"]),
        }
    return None


def selection_key(row: dict[str, Any]) -> tuple[float, float, int, str]:
    return (
        -float(row["best_val_accuracy"]),
        float(row["best_val_nll"]),
        int(row["best_epoch"]),
        str(row["id"]),
    )


def pending_report(stage: str, missing: list[str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": f"{stage}_pending",
        "selection_data": "validation_only",
        "missing": missing,
        "completed": rows,
    }


def build_report() -> dict[str, Any]:
    policy, _ = load_config("configs/sensitivity.yaml")
    primary = [row for identifier in PRIMARY_IDS if (row := completed_result(identifier))]
    missing_primary = [identifier for identifier in PRIMARY_IDS if not any(row["id"] == identifier for row in primary)]
    if missing_primary:
        return pending_report("primary", missing_primary, primary)

    primary_winner = min(primary, key=selection_key)
    threshold = float(policy["bounded_fallback"]["validation_accuracy_threshold"])
    optimization_candidates = list(primary)
    fallback: list[dict[str, Any]] = []
    if float(primary_winner["best_val_accuracy"]) < threshold:
        fallback = [row for identifier in FALLBACK_IDS if (row := completed_result(identifier))]
        missing_fallback = [identifier for identifier in FALLBACK_IDS if not any(row["id"] == identifier for row in fallback)]
        if missing_fallback:
            report = pending_report("fallback", missing_fallback, [*primary, *fallback])
            report["primary_winner"] = primary_winner
            report["fallback_triggered"] = True
            return report
        optimization_candidates.extend(fallback)

    optimization_winner = min(optimization_candidates, key=selection_key)
    normalized_id = f"{optimization_winner['id']}_norm"
    normalized = completed_result(normalized_id)
    if normalized is None:
        report = pending_report("normalization", [normalized_id], [*primary, *fallback])
        report.update(
            {
                "primary_winner": primary_winner,
                "fallback_triggered": bool(fallback),
                "optimization_winner": optimization_winner,
            }
        )
        return report

    final_winner = min((optimization_winner, normalized), key=selection_key)
    return {
        "status": "ready_to_lock",
        "selection_data": "validation_only",
        "selection_rule": policy["selection_rule"],
        "fallback_triggered": bool(fallback),
        "primary_results": primary,
        "fallback_results": fallback,
        "primary_winner": primary_winner,
        "optimization_winner": optimization_winner,
        "normalization_result": normalized,
        "final_winner": final_winner,
    }


def atomic_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    os.replace(temporary, path)


def write_locked_configs(report: dict[str, Any]) -> None:
    if report.get("status") != "ready_to_lock":
        raise RuntimeError("Sensitivity selection is not ready to lock")
    winner = report["final_winner"]
    selected, _ = load_config(run_config_path(winner["id"]))
    selected = copy.deepcopy(selected)
    selected["training"]["epochs"] = 50
    selected["experiment"]["name"] = "locked_table2_field_strength_resnet50"
    selected["experiment"]["additional_seeds"] = []
    selected["provenance"] = {
        "lock_status": "validation_selected_test_unseen",
        "selected_sensitivity_id": winner["id"],
        "selected_run_id": winner["run_id"],
        "selection_report": "reports/tables/sensitivity_selection.json",
    }
    lock_dir = PROJECT_ROOT / "configs" / "locked"
    hashes: dict[str, str] = {}
    for seed in (2025, 2026, 2027):
        locked = copy.deepcopy(selected)
        locked["experiment"]["seed"] = seed
        destination = lock_dir / f"seed{seed}.yaml"
        atomic_yaml(destination, locked)
        hashes[str(seed)] = config_sha256(locked)
    write_json(
        lock_dir / "LOCK.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit_before_lock": git_commit(PROJECT_ROOT),
            "selected_sensitivity_id": winner["id"],
            "selected_run_id": winner["run_id"],
            "test_status": "not_evaluated",
            "locked_config_sha256": hashes,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select sensitivity results using validation only and optionally lock seeds"
    )
    parser.add_argument("--write-lock", action="store_true")
    args = parser.parse_args()
    report = build_report()
    write_json(REPORT_PATH, report)
    print(json_dumps(report, indent=2))
    if args.write_lock:
        write_locked_configs(report)


if __name__ == "__main__":
    main()
