from pathlib import Path

from PIL import Image

from spurbreast_repro.data import discover_records, parse_spurbreast_filename


def _png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (8, 8), color=128).save(path)


def test_filename_parser_canonicalizes_patient_id() -> None:
    patient_id, slice_id = parse_spurbreast_filename("1-001-7.png")
    assert patient_id == "Breast_MRI_007"
    assert slice_id == "001"


def test_discovery_is_sorted_and_keeps_labels(tmp_path: Path) -> None:
    _png(tmp_path / "training" / "0" / "1-010-2.png")
    _png(tmp_path / "training" / "0" / "1-002-2.png")
    _png(tmp_path / "training" / "1" / "1-001-1.png")
    records = discover_records(
        tmp_path,
        "training",
        {"Breast_MRI_001": 1.5, "Breast_MRI_002": 3.0},
    )
    assert [record.label for record in records] == [0, 0, 1]
    assert [record.slice_id for record in records] == ["002", "010", "001"]
    assert {record.field_strength_t for record in records} == {1.5, 3.0}


def test_synthetic_patient_sets_are_disjoint(tmp_path: Path) -> None:
    for split, patient in (("training", 1), ("validation", 2), ("test", 3)):
        _png(tmp_path / split / "0" / f"1-001-{patient}.png")
        _png(tmp_path / split / "1" / f"1-002-{patient}.png")
    sets = {
        split: {record.patient_id for record in discover_records(tmp_path, split)}
        for split in ("training", "validation", "test")
    }
    assert sets["training"].isdisjoint(sets["validation"])
    assert sets["training"].isdisjoint(sets["test"])
    assert sets["validation"].isdisjoint(sets["test"])
