from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config_file(config_path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    config_path = config_path.resolve()
    if config_path in stack:
        chain = " -> ".join(str(item) for item in (*stack, config_path))
        raise ValueError(f"Configuration inheritance cycle: {chain}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a mapping in {config_path}")
    raw = dict(raw)
    parent_value = raw.pop("extends", None)
    if parent_value is None:
        return raw
    if not isinstance(parent_value, str):
        raise ValueError(f"extends must be a string in {config_path}")
    parent_path = Path(parent_value)
    if not parent_path.is_absolute():
        parent_path = config_path.parent / parent_path
    parent = _load_config_file(parent_path, (*stack, config_path))
    return _deep_merge(parent, raw)


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load a YAML config, recursively resolving optional ``extends`` parents."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()
    return _load_config_file(config_path, ()), config_path


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def config_sha256(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
