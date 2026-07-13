from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spurbreast_repro.baselines import (  # noqa: E402
    constant_predictions,
    field_strength_shortcut_predictions,
)
from spurbreast_repro.config import load_config, project_path  # noqa: E402
from spurbreast_repro.metrics import binary_metrics, patient_macro_metrics  # noqa: E402
from spurbreast_repro.utils import write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate non-learned shortcut baselines")
    parser.add_argument("--config", default="configs/reproduction.yaml")
    args = parser.parse_args()
    config, _ = load_config(args.config)
    manifest = project_path(config["data"]["manifest_file"])
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output = {}
    for split in ("training", "validation", "test"):
        selected = [row for row in rows if row["split"] == split]
        truth = [int(row["label"]) for row in selected]
        patients = [row["patient_id"] for row in selected]
        fields = [float(row["field_strength_t"]) for row in selected]
        field_predictions = field_strength_shortcut_predictions(fields)
        constant = constant_predictions(len(truth), label=0)
        output[split] = {
            "field_strength_oracle": {
                "slice_micro": binary_metrics(truth, field_predictions),
                "patient_macro": patient_macro_metrics(truth, field_predictions, patients),
            },
            "constant_non_tumor": {
                "slice_micro": binary_metrics(truth, constant),
                "patient_macro": patient_macro_metrics(truth, constant, patients),
            },
        }
    destination = PROJECT_ROOT / "reports" / "tables" / "shortcut_baselines.json"
    write_json(destination, output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
