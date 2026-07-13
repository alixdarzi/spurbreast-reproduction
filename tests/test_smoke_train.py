from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from spurbreast_repro.engine import evaluate_loader, train_one_epoch
from spurbreast_repro.utils import atomic_torch_save, json_dumps, write_json


class TinyDataset(Dataset):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int):
        image = torch.full((3, 8, 8), index / 8)
        label = index % 2
        metadata = {
            "patient_id": f"P{index // 2}",
            "slice_id": str(index),
            "relative_path": f"synthetic/{index}.png",
            "field_strength_t": 1.5 if index < 4 else 3.0,
        }
        return image, label, metadata


def test_one_training_and_evaluation_step_and_atomic_checkpoint(tmp_path: Path) -> None:
    loader = DataLoader(TinyDataset(), batch_size=4)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    training = train_one_epoch(
        model, loader, optimizer, criterion, torch.device("cpu"), scaler, max_batches=1
    )
    evaluation = evaluate_loader(
        model, loader, criterion, torch.device("cpu"), max_batches=1
    )
    assert training["n"] == 4
    assert evaluation["metrics"]["n"] == 4
    checkpoint = tmp_path / "latest.pt"
    atomic_torch_save({"model_state": model.state_dict()}, checkpoint)
    assert checkpoint.is_file()
    restored = torch.load(checkpoint, map_location="cpu")
    assert set(restored["model_state"]) == set(model.state_dict())
    strict_json = tmp_path / "metrics.json"
    write_json(strict_json, {"undefined_ppv": float("nan")})
    assert strict_json.read_text(encoding="utf-8").strip() == '{\n  "undefined_ppv": null\n}'
    assert json_dumps({"undefined_ppv": float("nan")}) == '{"undefined_ppv": null}'
