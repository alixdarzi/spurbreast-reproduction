from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spurbreast_repro.config import config_sha256, load_config, project_path  # noqa: E402
from spurbreast_repro.data import (  # noqa: E402
    SpurBreastDataset,
    build_transform,
    discover_records,
    load_field_map,
)
from spurbreast_repro.engine import evaluate_loader  # noqa: E402
from spurbreast_repro.losses import build_erm_loss  # noqa: E402
from spurbreast_repro.metrics import (  # noqa: E402
    patient_cluster_bootstrap,
    patient_macro_metrics,
    stratified_metrics,
)
from spurbreast_repro.models import build_model  # noqa: E402
from spurbreast_repro.utils import (  # noqa: E402
    json_dumps,
    load_trusted_checkpoint,
    seed_everything,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a locked checkpoint")
    parser.add_argument("--config", default="configs/reproduction.yaml")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", choices=("training", "validation", "test"), default=["training", "validation"])
    parser.add_argument("--allow-test", action="store_true", help="Explicit confirmation that model selection is locked")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if "test" in args.splits and not args.allow_test:
        raise SystemExit("Refusing test evaluation without --allow-test")
    config, _ = load_config(args.config)
    seed = int(config["experiment"]["seed"])
    seed_everything(seed, bool(config["experiment"].get("deterministic", True)))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device("cpu" if device_name == "auto" else device_name)
    checkpoint = load_trusted_checkpoint(args.checkpoint)
    if checkpoint["config_sha256"] != config_sha256(config):
        raise RuntimeError("Evaluation configuration does not match checkpoint")
    model_config = dict(config["model"])
    model_config["weights"] = "none"
    model = build_model(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    criterion = build_erm_loss()
    dataset_dir = project_path(config["data"]["dataset_dir"])
    manifest_parent = project_path(config["data"]["manifest_file"]).parent
    field_map = load_field_map(manifest_parent / "field_strength_by_patient.csv")
    output: dict[str, dict] = {}
    prediction_rows: list[dict] = []
    for split in args.splits:
        transform_split = "validation" if split == "training" else split
        transform = build_transform(
            transform_split,
            image_size=int(config["data"]["image_size"]),
            resize_size=int(config["data"]["resize_size"]),
            normalize=bool(config["data"]["normalize"]),
        )
        dataset = SpurBreastDataset(discover_records(dataset_dir, split, field_map), transform)
        loader = DataLoader(
            dataset,
            batch_size=int(config["data"]["batch_size"]),
            shuffle=False,
            num_workers=int(config["data"]["num_workers"]),
            pin_memory=device.type == "cuda",
        )
        raw = evaluate_loader(
            model,
            loader,
            criterion,
            device,
            amp=bool(config["training"].get("amp", False)) and device.type == "cuda",
            max_batches=config["training"].get("max_eval_batches"),
        )
        output[split] = {
            "slice_micro": raw["metrics"],
            "patient_macro": patient_macro_metrics(
                raw["y_true"], raw["y_pred"], raw["patient_ids"]
            ),
            "field_strength_stratified": stratified_metrics(
                raw["y_true"], raw["y_pred"], raw["probabilities"], raw["field_strengths"]
            ),
            "patient_cluster_95ci": patient_cluster_bootstrap(
                raw["y_true"],
                raw["y_pred"],
                raw["patient_ids"],
                resamples=int(config["evaluation"]["bootstrap_resamples"]),
                seed=seed,
            ),
        }
        for index in range(len(raw["y_true"])):
            prediction_rows.append(
                {
                    "split": split,
                    "relative_path": raw["relative_paths"][index],
                    "patient_id": raw["patient_ids"][index],
                    "field_strength_t": raw["field_strengths"][index],
                    "y_true": raw["y_true"][index],
                    "y_pred": raw["y_pred"][index],
                    "positive_probability": raw["probabilities"][index],
                }
            )
    result_dir = project_path(config["paths"]["results"]) / args.checkpoint.resolve().parent.name
    result_dir.mkdir(parents=True, exist_ok=True)
    write_json(result_dir / "evaluation.json", output)
    with (result_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    print(json_dumps(output, indent=2))


if __name__ == "__main__":
    main()
