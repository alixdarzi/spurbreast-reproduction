from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torchvision


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spurbreast_repro.utils import environment_summary  # noqa: E402


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = environment_summary(device)
    summary["torchvision"] = torchvision.__version__
    summary["project_root"] = str(PROJECT_ROOT)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
