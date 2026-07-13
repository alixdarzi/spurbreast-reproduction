from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_erm_loss() -> nn.Module:
    """The documented primary assumption: unweighted cross-entropy."""
    return nn.CrossEntropyLoss()


class GroupDROLoss(nn.Module):
    """Minimal GroupDRO objective reserved for the separately labelled extension."""

    def __init__(self, num_groups: int, step_size: float = 0.01) -> None:
        super().__init__()
        if num_groups < 1:
            raise ValueError("num_groups must be positive")
        self.step_size = step_size
        self.register_buffer("group_weights", torch.ones(num_groups) / num_groups)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, groups: torch.Tensor):
        losses = F.cross_entropy(logits, targets, reduction="none")
        group_losses = []
        present = []
        for group_index in range(len(self.group_weights)):
            mask = groups == group_index
            if mask.any():
                group_losses.append(losses[mask].mean())
                present.append(group_index)
        if not present:
            raise ValueError("No valid groups in batch")
        stacked = torch.stack(group_losses)
        with torch.no_grad():
            indices = torch.as_tensor(present, device=self.group_weights.device)
            self.group_weights[indices] *= torch.exp(self.step_size * stacked.detach())
            self.group_weights /= self.group_weights.sum()
        weights = self.group_weights[torch.as_tensor(present, device=self.group_weights.device)]
        weights = weights / weights.sum()
        return torch.sum(weights * stacked)
