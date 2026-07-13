from pathlib import Path

import pytest

from spurbreast_repro.config import config_sha256, load_config, project_path


def test_reproduction_config_is_stable_and_test_is_not_a_training_input() -> None:
    config, _ = load_config("configs/reproduction.yaml")
    assert config["data"]["batch_size"] == 32
    assert config["training"]["epochs"] == 50
    assert config["data"]["normalize"] is False
    assert "test" not in config["training"]
    assert len(config_sha256(config)) == 64
    assert project_path(config["data"]["dataset_dir"]).is_absolute()


def test_config_inheritance_deep_merges_without_retaining_extends(tmp_path: Path) -> None:
    parent = tmp_path / "parent.yaml"
    parent.write_text("model:\n  name: resnet50\n  classes: 2\ntraining:\n  epochs: 50\n", encoding="utf-8")
    child = tmp_path / "child.yaml"
    child.write_text("extends: parent.yaml\ntraining:\n  epochs: 10\n", encoding="utf-8")
    config, path = load_config(child)
    assert path == child.resolve()
    assert config == {
        "model": {"name": "resnet50", "classes": 2},
        "training": {"epochs": 10},
    }


def test_config_inheritance_cycle_is_rejected(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("extends: second.yaml\n", encoding="utf-8")
    second.write_text("extends: first.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inheritance cycle"):
        load_config(first)
