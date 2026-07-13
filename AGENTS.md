# AGENTS.md

## Approved scientific target

Reproduce the ResNet-50 row for **Magnetic Field Strength** in Table 2 of
*SpurBreast: A Curated Dataset for Investigating Spurious Correlations in
Real-world Breast MRI Classification* (MICCAI 2025).

The task is 2D slice classification among patients with biopsy-confirmed
invasive breast cancer: tumor-containing slice (`1`) versus non-tumor slice
(`0`). It is not patient-level cancer diagnosis.

The approved reproduction category is **close reproduction**. The primary
scientific claim is the field-strength shortcut produced by patient-disjoint,
deliberately confounded training and validation splits and an unbiased test
split.

## Non-negotiable protocol rules

- Use the official `field_strength.zip` release and verify its MD5 checksum.
- Keep `data/raw/` immutable after download and extraction.
- Never place a patient in more than one of training, validation, and test.
- Preserve the published relationship: training/validation tumor slices are
  from 1.5 T patients and non-tumor slices are from 3 T patients.
- Do not tune hyperparameters, thresholds, calibration, or preprocessing on
  the test set.
- Select checkpoints using validation accuracy only. Resolve ties by lower
  validation NLL, then the earliest epoch.
- Report every prespecified seed. Never choose a seed using test performance.
- Primary Table 2 metrics use one global slice-level confusion matrix. Report
  patient-macro results separately.
- Record every deviation affecting data, splits, model, preprocessing,
  metrics, or reproduction category in `docs/DECISIONS.md` before running it.
- Do not commit medical images, raw metadata, generated manifests containing
  patient identifiers, checkpoints, credentials, or local absolute paths.

## Repository conventions

- English only for code, documentation, configurations, and results.
- Source code lives in `src/spurbreast_repro/`; notebooks call this package.
- One frozen YAML configuration and one unique run ID per substantive run.
- Generated artifacts go under `results/`, `reports/`, `checkpoints/`, or
  `experiments/` and must retain provenance.
- Use `pathlib` and project-relative paths. Do not hard-code a user directory.
- Keep failed runs in the experiment registry with an explanation.
- Add or update tests when fixing a scientific or data-integrity defect.

## Canonical commands

```powershell
python scripts/download_data.py --config configs/reproduction.yaml
python scripts/prepare_data.py --config configs/reproduction.yaml --verify-images --hash-images
python -m pytest
python scripts/run_or_resume.py --config configs/sensitivity_runs/H1.yaml --device cuda
python scripts/select_sensitivity.py
```

For the required resume check, first run the smoke configuration with
`--stop-after-epoch 0`, then pass its `latest.pt` to `--resume`.

After H1–H4, follow the selector's requested fallback/normalization run. Only
use `select_sensitivity.py --write-lock` when its status is `ready_to_lock`.
Commit the generated lock before executing any final seed or test evaluation.

## Verification required before expensive training

- Archive checksum and safe extraction pass.
- Expected image and patient counts match the published split.
- Cross-split patient intersections are empty.
- All test patients occur in both classes.
- Field-strength metadata join has no missing patients.
- Preprocessing visualization and tensor-range checks pass.
- Unit tests and reduced smoke training pass.
- Checkpoint save/resume reproduces the next training step.

## Data and licensing safety

The upstream TCIA and SpurBreast data are CC BY-NC 4.0. Do not redistribute
the dataset. Repository code is original unless attribution states otherwise.
Do not publish trained weights until their redistribution status has been
reviewed. This project is research-only and not for clinical use.
