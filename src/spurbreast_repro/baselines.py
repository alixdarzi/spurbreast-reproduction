from __future__ import annotations

from typing import Sequence


def field_strength_shortcut_predictions(
    field_strengths: Sequence[float], positive_field_t: float = 1.5
) -> list[int]:
    """Paper-observed shortcut: predict tumor for 1.5 T and non-tumor for 3 T."""
    return [int(abs(float(value) - positive_field_t) < 1e-6) for value in field_strengths]


def constant_predictions(n: int, label: int = 0) -> list[int]:
    if label not in (0, 1):
        raise ValueError("Constant binary label must be 0 or 1")
    return [label] * n
