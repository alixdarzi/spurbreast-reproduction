from spurbreast_repro.config import config_sha256, load_config, project_path


def test_reproduction_config_is_stable_and_test_is_not_a_training_input() -> None:
    config, _ = load_config("configs/reproduction.yaml")
    assert config["data"]["batch_size"] == 32
    assert config["training"]["epochs"] == 50
    assert config["data"]["normalize"] is False
    assert "test" not in config["training"]
    assert len(config_sha256(config)) == 64
    assert project_path(config["data"]["dataset_dir"]).is_absolute()
