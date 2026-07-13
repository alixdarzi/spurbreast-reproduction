from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
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
from spurbreast_repro.engine import evaluate_loader, train_one_epoch  # noqa: E402
from spurbreast_repro.losses import build_erm_loss  # noqa: E402
from spurbreast_repro.models import build_model  # noqa: E402
from spurbreast_repro.utils import (  # noqa: E402
    append_jsonl,
    atomic_torch_save,
    capture_rng_state,
    environment_summary,
    git_commit,
    restore_rng_state,
    seed_everything,
    seed_worker,
    utc_now,
    write_json,
)


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]):
    name = config["optimizer"].lower()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    common = {
        "lr": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
    }
    if name == "adamw":
        return torch.optim.AdamW(parameters, **common)
    if name == "adam":
        return torch.optim.Adam(parameters, **common)
    if name == "sgd":
        return torch.optim.SGD(parameters, momentum=float(config.get("momentum", 0.0)), **common)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer, config: dict[str, Any]):
    name = config["scheduler"].lower()
    if name == "constant":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(config["epochs"]), eta_min=0.0
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def append_registry(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id",
        "started_at",
        "finished_at",
        "status",
        "config_path",
        "config_sha256",
        "git_commit",
        "dataset_md5",
        "seed",
        "device",
        "runtime_seconds",
        "best_epoch",
        "best_val_accuracy",
        "result_path",
        "notes",
    ]
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def make_loaders(config: dict[str, Any], seed: int):
    data_config = config["data"]
    dataset_dir = project_path(data_config["dataset_dir"])
    field_map_path = project_path(data_config["manifest_file"]).parent / "field_strength_by_patient.csv"
    field_map = load_field_map(field_map_path)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loaders = {}
    for split in ("training", "validation"):
        transform = build_transform(
            split,
            image_size=int(data_config["image_size"]),
            resize_size=int(data_config["resize_size"]),
            normalize=bool(data_config["normalize"]),
        )
        dataset = SpurBreastDataset(discover_records(dataset_dir, split, field_map), transform)
        workers = int(data_config["num_workers"])
        loaders[split] = DataLoader(
            dataset,
            batch_size=int(data_config["batch_size"]),
            shuffle=split == "training",
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            generator=generator,
            persistent_workers=workers > 0,
        )
    return loaders, generator


def checkpoint_payload(
    *,
    epoch: int,
    config: dict[str, Any],
    model,
    optimizer,
    scheduler,
    scaler,
    generator,
    best_epoch: int,
    best_accuracy: float,
    best_nll: float,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "config": config,
        "config_sha256": config_sha256(config),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "loader_generator_state": generator.get_state(),
        "rng_state": capture_rng_state(),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_accuracy,
        "best_val_nll": best_nll,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train without touching the final test split")
    parser.add_argument("--config", default="configs/smoke_test.yaml")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    seed = int(config["experiment"]["seed"])
    seed_everything(seed, bool(config["experiment"].get("deterministic", True)))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device_name = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    device = torch.device(device_name)

    run_started = utc_now()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{config['experiment']['name']}-seed{seed}-{timestamp}"
    if args.resume:
        run_id = args.resume.resolve().parent.name
    checkpoint_dir = project_path(config["paths"]["checkpoints"]) / run_id
    result_dir = project_path(config["paths"]["results"]) / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    write_json(result_dir / "environment.json", environment_summary(device))

    checksum_file = PROJECT_ROOT / "data" / "raw" / "CHECKSUMS.json"
    dataset_md5 = "unverified"
    if checksum_file.exists():
        dataset_md5 = json.loads(checksum_file.read_text(encoding="utf-8"))["archive"]["md5"]
    registry = project_path(config["paths"]["registry"])
    base_registry = {
        "run_id": run_id,
        "started_at": run_started,
        "config_path": config_path.relative_to(PROJECT_ROOT).as_posix(),
        "config_sha256": config_sha256(config),
        "git_commit": git_commit(PROJECT_ROOT),
        "dataset_md5": dataset_md5,
        "seed": seed,
        "device": str(device),
    }
    append_registry(registry, {**base_registry, "status": "running"})
    start_time = time.monotonic()
    try:
        loaders, generator = make_loaders(config, seed)
        model = build_model(**config["model"]).to(device)
        criterion = build_erm_loss()
        optimizer = build_optimizer(model, config["training"])
        scheduler = build_scheduler(optimizer, config["training"])
        amp_enabled = bool(config["training"].get("amp", False)) and device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        first_epoch = 0
        best_epoch = -1
        best_accuracy = float("-inf")
        best_nll = float("inf")
        if args.resume:
            state = torch.load(args.resume, map_location=device)
            if state["config_sha256"] != config_sha256(config):
                raise RuntimeError("Resume checkpoint configuration does not match")
            model.load_state_dict(state["model_state"])
            optimizer.load_state_dict(state["optimizer_state"])
            if scheduler is not None and state["scheduler_state"] is not None:
                scheduler.load_state_dict(state["scheduler_state"])
            scaler.load_state_dict(state["scaler_state"])
            generator.set_state(state["loader_generator_state"])
            restore_rng_state(state["rng_state"])
            first_epoch = int(state["epoch"]) + 1
            best_epoch = int(state["best_epoch"])
            best_accuracy = float(state["best_val_accuracy"])
            best_nll = float(state["best_val_nll"])

        history_path = result_dir / "history.jsonl"
        for epoch in range(first_epoch, int(config["training"]["epochs"])):
            train_metrics = train_one_epoch(
                model,
                loaders["training"],
                optimizer,
                criterion,
                device,
                scaler,
                amp=amp_enabled,
                max_batches=config["training"].get("max_train_batches"),
            )
            validation = evaluate_loader(
                model,
                loaders["validation"],
                criterion,
                device,
                amp=amp_enabled,
                max_batches=config["training"].get("max_eval_batches"),
            )["metrics"]
            if scheduler is not None:
                scheduler.step()
            accuracy = float(validation["accuracy"])
            nll = float(validation["nll"])
            improved = accuracy > best_accuracy + 1e-12 or (
                abs(accuracy - best_accuracy) <= 1e-12 and nll < best_nll - 1e-12
            )
            if improved:
                best_epoch, best_accuracy, best_nll = epoch, accuracy, nll
            payload = checkpoint_payload(
                epoch=epoch,
                config=config,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                generator=generator,
                best_epoch=best_epoch,
                best_accuracy=best_accuracy,
                best_nll=best_nll,
            )
            atomic_torch_save(payload, checkpoint_dir / "latest.pt")
            if improved:
                atomic_torch_save(payload, checkpoint_dir / "best.pt")
            row = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "validation": validation,
                "best_epoch": best_epoch,
            }
            append_jsonl(history_path, row)
            print(json.dumps(row, sort_keys=True))

        runtime = time.monotonic() - start_time
        summary = {
            "run_id": run_id,
            "status": "completed",
            "best_epoch": best_epoch,
            "best_val_accuracy": best_accuracy,
            "best_val_nll": best_nll,
            "runtime_seconds": runtime,
        }
        write_json(result_dir / "summary.json", summary)
        append_registry(
            registry,
            {
                **base_registry,
                "finished_at": utc_now(),
                "status": "completed",
                "runtime_seconds": f"{runtime:.3f}",
                "best_epoch": best_epoch,
                "best_val_accuracy": best_accuracy,
                "result_path": result_dir.relative_to(PROJECT_ROOT).as_posix(),
            },
        )
    except Exception as error:
        runtime = time.monotonic() - start_time
        append_registry(
            registry,
            {
                **base_registry,
                "finished_at": utc_now(),
                "status": "failed",
                "runtime_seconds": f"{runtime:.3f}",
                "notes": repr(error),
            },
        )
        raise


if __name__ == "__main__":
    main()
