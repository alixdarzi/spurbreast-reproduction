from pathlib import Path

from spurbreast_repro.data import SliceRecord, balanced_record_prefix


def make_record(label: int, index: int) -> SliceRecord:
    return SliceRecord(
        path=Path(f"{label}-{index}.png"),
        relative_path=f"validation/{label}/{index}.png",
        split="validation",
        label=label,
        patient_id=f"P{label}{index}",
        slice_id=str(index),
        field_strength_t=1.5 if label else 3.0,
    )


def test_balanced_record_prefix_interleaves_labels() -> None:
    records = [make_record(0, index) for index in range(8)] + [
        make_record(1, index) for index in range(8)
    ]
    selected = balanced_record_prefix(records, total=8)
    assert [record.label for record in selected] == [0, 1] * 4
