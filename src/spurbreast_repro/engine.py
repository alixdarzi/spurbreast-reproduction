from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

from .metrics import binary_metrics


def train_one_epoch(
    model: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    amp: bool = False,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_examples = 0
    correct = 0
    for batch_index, (images, labels, _) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = labels.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_examples += batch_size
        correct += int((logits.argmax(dim=1) == labels).sum())
    if total_examples == 0:
        raise RuntimeError("Training loader yielded no examples")
    return {
        "loss": total_loss / total_examples,
        "accuracy": correct / total_examples,
        "n": float(total_examples),
    }


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    device: torch.device,
    amp: bool = False,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[float] = []
    patient_ids: list[str] = []
    field_strengths: list[float] = []
    relative_paths: list[str] = []
    for batch_index, (images, labels, metadata) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)
        positive_probability = torch.softmax(logits, dim=1)[:, 1]
        prediction = logits.argmax(dim=1)
        batch_size = labels.shape[0]
        total_loss += float(loss) * batch_size
        total_examples += batch_size
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(prediction.cpu().tolist())
        probabilities.extend(positive_probability.cpu().tolist())
        patient_ids.extend(list(metadata["patient_id"]))
        relative_paths.extend(list(metadata["relative_path"]))
        fields = metadata["field_strength_t"]
        field_strengths.extend(fields.cpu().tolist() if torch.is_tensor(fields) else list(fields))
    if total_examples == 0:
        raise RuntimeError("Evaluation loader yielded no examples")
    metrics = binary_metrics(y_true, y_pred, probabilities)
    metrics["loss"] = total_loss / total_examples
    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "probabilities": probabilities,
        "patient_ids": patient_ids,
        "field_strengths": field_strengths,
        "relative_paths": relative_paths,
    }
