from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


def build_model(
    architecture: str = "resnet50",
    weights: str = "IMAGENET1K_V1",
    num_classes: int = 2,
    freeze_backbone: bool = False,
) -> nn.Module:
    if architecture.lower() != "resnet50":
        raise ValueError(f"Unsupported architecture: {architecture}")
    normalized_weights = weights.strip().upper()
    if normalized_weights in {"NONE", "NULL", "RANDOM"}:
        resolved_weights = None
    elif normalized_weights == "IMAGENET1K_V1":
        resolved_weights = ResNet50_Weights.IMAGENET1K_V1
    else:
        raise ValueError(f"Unsupported ResNet-50 weights: {weights}")
    model = resnet50(weights=resolved_weights)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
