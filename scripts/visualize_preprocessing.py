from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spurbreast_repro.config import load_config, project_path  # noqa: E402
from spurbreast_repro.data import build_transform, discover_records  # noqa: E402


def tensor_image(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an ignored local preprocessing audit")
    parser.add_argument("--config", default="configs/reproduction.yaml")
    parser.add_argument(
        "--output", default="reports/figures/preprocessing_examples.png"
    )
    args = parser.parse_args()
    config, _ = load_config(args.config)
    dataset_dir = project_path(config["data"]["dataset_dir"])
    records = discover_records(dataset_dir, "training")
    examples = [next(record for record in records if record.label == label) for label in (0, 1)]
    train_transform = build_transform(
        "training",
        image_size=int(config["data"]["image_size"]),
        resize_size=int(config["data"]["resize_size"]),
        normalize=bool(config["data"]["normalize"]),
    )
    eval_transform = build_transform(
        "validation",
        image_size=int(config["data"]["image_size"]),
        resize_size=int(config["data"]["resize_size"]),
        normalize=bool(config["data"]["normalize"]),
    )
    figure, axes = plt.subplots(2, 3, figsize=(10, 7))
    torch.manual_seed(int(config["experiment"]["seed"]))
    for row, record in enumerate(examples):
        with Image.open(record.path) as source:
            image = source.convert("RGB")
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"Official PNG — class {record.label}")
        axes[row, 1].imshow(tensor_image(train_transform(image)))
        axes[row, 1].set_title("Training transform")
        axes[row, 2].imshow(tensor_image(eval_transform(image)))
        axes[row, 2].set_title("Evaluation transform")
        for axis in axes[row]:
            axis.axis("off")
    figure.suptitle("Local audit only — do not commit medical-image derivatives")
    figure.tight_layout()
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
