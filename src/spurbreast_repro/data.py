from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


SPLITS = ("training", "validation", "test")
FILENAME_PATTERN = re.compile(
    r"^(?P<prefix>[^-]+)-(?P<slice_id>[^-]+)-(?P<patient_token>\d+)\.png$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class SliceRecord:
    path: Path
    relative_path: str
    split: str
    label: int
    patient_id: str
    slice_id: str
    field_strength_t: float | None = None


def parse_spurbreast_filename(path: str | Path) -> tuple[str, str]:
    """Return canonical TCIA patient ID and slice ID from a SpurBreast PNG."""
    name = Path(path).name
    match = FILENAME_PATTERN.match(name)
    if match is None:
        raise ValueError(f"Unexpected SpurBreast filename: {name}")
    patient_id = f"Breast_MRI_{int(match.group('patient_token')):03d}"
    return patient_id, match.group("slice_id")


def load_field_map(path: str | Path) -> dict[str, float]:
    """Load a generated patient-to-field-strength CSV."""
    mapping: dict[str, float] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            mapping[row["patient_id"]] = float(row["field_strength_t"])
    return mapping


def discover_records(
    dataset_dir: str | Path,
    split: str,
    field_map: Mapping[str, float] | None = None,
) -> list[SliceRecord]:
    """Discover an official split in deterministic path order."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    dataset_dir = Path(dataset_dir)
    records: list[SliceRecord] = []
    for label in (0, 1):
        label_dir = dataset_dir / split / str(label)
        if not label_dir.is_dir():
            raise FileNotFoundError(f"Missing split folder: {label_dir}")
        for path in sorted(label_dir.glob("*.png"), key=lambda item: item.name):
            patient_id, slice_id = parse_spurbreast_filename(path)
            field_strength = None if field_map is None else field_map.get(patient_id)
            records.append(
                SliceRecord(
                    path=path.resolve(),
                    relative_path=path.relative_to(dataset_dir).as_posix(),
                    split=split,
                    label=label,
                    patient_id=patient_id,
                    slice_id=slice_id,
                    field_strength_t=field_strength,
                )
            )
    return records


def build_transform(
    split: str,
    image_size: int = 224,
    resize_size: int = 256,
    normalize: bool = False,
) -> Callable:
    """Build repository-documented transforms, with optional sensitivity normalization."""
    operations: list[Callable]
    if split == "training":
        operations = [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomResizedCrop(image_size),
            transforms.ToTensor(),
        ]
    elif split in ("validation", "test"):
        operations = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    else:
        raise ValueError(f"Unknown split: {split}")
    if normalize:
        operations.append(
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        )
    return transforms.Compose(operations)


class SpurBreastDataset(Dataset):
    def __init__(self, records: Iterable[SliceRecord], transform: Callable) -> None:
        self.records = list(records)
        if not self.records:
            raise ValueError("Dataset has no records")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.path) as source:
            image = source.convert("RGB")
        image = self.transform(image)
        metadata = {
            "patient_id": record.patient_id,
            "slice_id": record.slice_id,
            "relative_path": record.relative_path,
            "field_strength_t": (
                float("nan") if record.field_strength_t is None else record.field_strength_t
            ),
        }
        return image, record.label, metadata
