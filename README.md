# Breast MRI Field-Strength Bias

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alixdarzi/breast-mri-field-strength-bias/blob/main/notebooks/colab_reproduction.ipynb)
[![License: MIT](https://img.shields.io/badge/Code-MIT-2ea44f.svg)](LICENSE)
[![Python 3.10–3.12](https://img.shields.io/badge/Python-3.10–3.12-3776ab.svg)](pyproject.toml)

A reproducible analysis of magnetic-field-strength shortcut learning in breast
MRI slice classification, based on the **Magnetic Field Strength — ResNet-50**
experiment in Table 2 of the SpurBreast paper (MICCAI 2025).

## Main result

The locked three-seed reproduction obtained **0.5175 ± 0.0060 test accuracy**,
compared with **0.52** in the paper. Performance fell from 0.9893 validation
accuracy to near chance when the field-strength correlation was broken.

![Published Table 2 versus the locked reproduction](reports/final_results/table2_comparison.png)

| Split | Accuracy | PPV | NPV |
|---|---:|---:|---:|
| Training | 0.9998 ± 0.0003 | 0.9995 ± 0.0007 | 1.0000 ± 0.0000 |
| Validation | 0.9893 ± 0.0020 | 0.9957 ± 0.0032 | 0.9831 ± 0.0067 |
| Test | **0.5175 ± 0.0060** | 0.5215 ± 0.0073 | 0.5148 ± 0.0051 |

On test, the mean predicted-positive rate was **0.8735 at 1.5 T** and
**0.0200 at 3 T**. This large 0.8535 gap provides direct evidence that the
network retained the magnetic-field shortcut.

## Clinical task

All patients in the source cohort have biopsy-confirmed invasive breast
cancer. The model classifies individual 2D MRI slices as:

- tumor-containing; or
- non-tumor.

This is not breast-cancer screening, patient-level diagnosis, tumor detection,
or a clinical decision system.

## Experimental design

- Official `field_strength.zip` release, checksum verified.
- 19,926 grayscale MRI slices from 700 patients.
- Patient-disjoint training, validation, and test sets.
- Deliberately confounded training/validation data: tumor slices from 1.5 T
  patients and non-tumor slices from separate 3 T patients.
- Unbiased test patients contain both slice labels.
- ImageNet-pretrained ResNet-50, SGD, cosine schedule, and ImageNet
  normalization.
- Three prespecified 50-epoch seeds: 2025, 2026, and 2027.
- Validation-only checkpoint selection; no test-set tuning.
- Slice-micro, patient-macro, patient-cluster bootstrap, field-stratified, and
  calibration analyses.

The exact protocol and evidence provenance are documented in
[the reproduction plan](docs/REPRODUCTION_PLAN.md) and
[traceability table](docs/TRACEABILITY.md).

## Reproduce the analysis

The Colab notebook is the recommended entry point:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alixdarzi/breast-mri-field-strength-bias/blob/main/notebooks/colab_reproduction.ipynb)

For a local CPU audit environment:

```powershell
python -m pip install -r requirements-local-cpu.txt
python -m pip install -e .
python -m pytest
python scripts/download_data.py --config configs/reproduction.yaml
python scripts/prepare_data.py --config configs/reproduction.yaml --verify-images --hash-images
```

After locked evaluations exist, regenerate all public tables and figures with:

```powershell
python scripts/summarize_results.py
```

The medical images are not stored in this repository. The upstream data are
CC BY-NC 4.0 and must be obtained from the official release.

## Results and documentation

- [Final report](reports/FINAL_REPORT.md)
- [Table 2 comparison](reports/final_results/table2_comparison.csv)
- [Field-strength metrics](reports/final_results/field_strength_metrics.csv)
- [Calibration summary](reports/final_results/calibration_summary.csv)
- [Patient-cluster intervals](reports/final_results/patient_cluster_intervals.csv)
- [Data audit](docs/DATA_AUDIT.md)
- [Decisions and assumptions](docs/DECISIONS.md)
- [Known limitations](docs/LIMITATIONS.md)
- [Optional extensions](docs/EXTENSIONS.md)

Raw predictions, patient identifiers, medical images, and checkpoints remain
private and are excluded from Git.

## Interpretation limitation

Test accuracy closely reproduces Table 2, but the published balanced-test
values 0.52 accuracy, 0.62 PPV, and 0.41 NPV cannot all come from one standard
global confusion matrix. This repository reports standard pooled slice metrics
and labels the PPV/NPV difference as an unresolved metric-provenance issue.

## Citation

See [CITATION.cff](CITATION.cff) and the original paper:
[SpurBreast: A Curated Dataset for Investigating Spurious Correlations in
Real-world Breast MRI Classification](https://doi.org/10.1007/978-3-032-05325-1_53).

## License and medical disclaimer

The original code is MIT licensed. Dataset files retain their original
CC BY-NC 4.0 terms and are not redistributed.

**Research use only. This project is not a medical device and must not be used
for clinical diagnosis, screening, triage, or patient management.**
