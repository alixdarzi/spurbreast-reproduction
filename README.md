# SpurBreast Field-Strength Reproduction

Close reproduction of the **Magnetic Field Strength — ResNet-50** row in
Table 2 of *SpurBreast: A Curated Dataset for Investigating Spurious
Correlations in Real-world Breast MRI Classification* (MICCAI 2025).

## Clinical and scientific scope

All source patients have biopsy-confirmed invasive breast cancer. The model
classifies individual 2D MRI slices as tumor-containing or non-tumor. This is
not breast-cancer screening, patient-level diagnosis, tumor detection, or a
clinical decision system.

The central experiment deliberately makes magnetic field strength a perfect
shortcut in training and validation: tumor slices come from 1.5 T patients,
while non-tumor slices come from different 3 T patients. The patient-disjoint
test set contains both slice labels for every patient and therefore breaks the
shortcut.

## Target values

| Split | Accuracy | PPV | NPV |
|---|---:|---:|---:|
| Training | 0.99 | 0.98 | 0.99 |
| Validation | 0.99 | 1.00 | 0.98 |
| Test | 0.52 | 0.62 | 0.41 |

The released test split is exactly class-balanced, but the three published
test metrics cannot all arise from one standard global confusion matrix.
This repository therefore reports globally pooled slice metrics and separately
labelled patient-macro metrics, with a transparent discrepancy analysis.

## Project status

The checksum-verified data audit is complete and passed. Full reproduction
results will be added only after tests, the checkpoint/resume smoke run,
Approval Gate 3, and the final GPU runs. No paper result is claimed as
reproduced yet.

## Data

- SpurBreast dataset: https://zenodo.org/records/17128791
- Required archive: `field_strength.zip` (905,891,386 bytes)
- Expected MD5: `dbe61da7dc7b06c69c10dbbea0a13b40`
- Upstream TCIA collection:
  https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/
- Terms: CC BY-NC 4.0; data are never committed to Git.

Expected published split:

| Split | Images | Unique patients |
|---|---:|---:|
| Training | 9,562 | 400 |
| Validation | 3,576 | 150 |
| Test | 6,788 | 150 |

The release was fully decoded and hashed: all 19,926 files are valid 320×320
grayscale PNGs, with zero duplicate hashes and zero cross-split patient
overlap. See `docs/DATA_AUDIT.md` and `data/README.md`.

## Setup

Python 3.11 is the supported local environment. The verified Phase 4 smoke
environment is CPU-only:

```powershell
python -m pip install -r requirements-local-cpu.txt
python -m pip install -e .
python -m pytest
```

`requirements-local-cu117.txt` is provided for the GTX 1660 Ti's current CUDA
11.7-compatible driver, but its large wheel is optional because final training
is assigned to Colab. Colab must use `requirements-colab.txt` so its existing
CUDA-enabled PyTorch build is not replaced by a CPU wheel. Every run records
the actual Torch, CUDA, and GPU versions.

## Canonical workflow

```powershell
python scripts/download_data.py --config configs/reproduction.yaml
python scripts/prepare_data.py --config configs/reproduction.yaml --verify-images --hash-images
python -m pytest
python scripts/train.py --config configs/smoke_test.yaml
python scripts/train.py --config configs/reproduction.yaml
python scripts/evaluate.py --config configs/reproduction.yaml --checkpoint checkpoints/<run-id>/best.pt --splits training validation test --allow-test
```

Hyperparameter sensitivity runs are validation-only. The final test set is not
used until preprocessing, optimization, checkpoint selection, and seeds have
been locked.

## Repository layout

- `configs/`: frozen experiment configurations
- `data/`: download documentation and ignored local data
- `docs/`: paper summary, protocol, traceability, decisions, limitations
- `src/spurbreast_repro/`: reusable implementation
- `scripts/`: command-line entry points
- `tests/`: split, preprocessing, metric, and smoke tests
- `notebooks/`: audit and presentation notebooks that call source code
- `experiments/`: open experiment registry
- `reports/`: generated tables, figures, and final report

## Reproducibility safeguards

- patient-disjoint split assertions;
- deterministic file ordering and seeded data loaders;
- configuration hashes and unique run IDs;
- atomic best/latest checkpoints with RNG state;
- global and patient-clustered statistical reporting;
- no test-set hyperparameter or threshold selection;
- preservation of failed and invalidated runs.

## Citation

See `CITATION.cff` and the original paper:
https://doi.org/10.1007/978-3-032-05325-1_53

## License and disclaimer

Original code in this repository is MIT licensed. Dataset files remain under
their original CC BY-NC 4.0 terms and are not redistributed.

**Research use only. This project is not a medical device and must not be used
for clinical diagnosis, screening, triage, or patient management.**
