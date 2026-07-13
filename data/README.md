# Data workflow

Data are downloaded from official sources and remain outside Git.

## Required files

| File | Source | Integrity |
|---|---|---|
| `data/raw/field_strength.zip` | Zenodo record 17128791 | MD5 `dbe61da7dc7b06c69c10dbbea0a13b40` |
| `data/raw/Clinical_and_Other_Features.xlsx` | TCIA Duke Breast Cancer MRI | SHA-256 recorded after download |

The archive extracts to `data/raw/field_strength/` with `training`,
`validation`, and `test` folders, each containing class folders `0` and `1`.

## Commands

```powershell
python scripts/download_data.py --config configs/reproduction.yaml
python scripts/prepare_data.py --config configs/reproduction.yaml --verify-images --hash-images
```

The preparation step writes an ignored internal manifest under
`data/interim/` and a de-identified aggregate audit under `reports/tables/`.
It does not modify the raw archive or extracted PNG files.

## Expected audit invariants

- Training: 9,562 images, 400 patients.
- Validation: 3,576 images, 150 patients.
- Test: 6,788 images, 150 patients.
- Zero patient overlap between splits.
- Training/validation label `1` patients are exclusively 1.5 T.
- Training/validation label `0` patients are exclusively 3 T.
- Every test patient occurs in both label folders.
- Test patients: 71 at 1.5 T and 79 at 3 T.

## Safety

Treat `data/raw/` as immutable. Never commit images, spreadsheets, or internal
manifests. The source data are licensed CC BY-NC 4.0 and are for research use.
