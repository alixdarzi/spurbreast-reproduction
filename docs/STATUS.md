# Project checkpoint and resume status

Last updated: 2026-07-17 (Asia/Tehran)

## Current state

The primary close reproduction is complete. All three locked ResNet-50 runs
(seeds 2025, 2026, and 2027) finished 50 epochs, their validation-selected
checkpoints were evaluated once on training, validation, and test, and the
de-identified aggregate tables and figures were generated. The private
per-slice prediction files and checkpoints remain on Google Drive.

Primary test accuracy is **0.5175 ± 0.0060**, compared with 0.52 in Table 2.
Validation accuracy is 0.9893 ± 0.0020, producing a 0.4717
validation-to-test drop. Test predicted-positive rates are 0.8735 at 1.5 T and
0.0200 at 3 T. These results meet the prespecified close-reproduction criteria
and reproduce the field-strength shortcut collapse.

The published test PPV/NPV values are not reproduced: the global results are
0.5215/0.5148 versus 0.62/0.41. This is retained as a metric-provenance
limitation because the three paper values cannot coexist in one standard
confusion matrix on the released balanced test set.

## Completed stages

- Full paper, official repository, and dataset evidence review.
- Reproduction protocol, traceability table, assumptions, and compute plan.
- Checksum-verified 19,926-image audit with 700 patients and no leakage.
- Preprocessing inspection, tests, smoke training, and checkpoint/resume test.
- Validation-only sensitivity screen and committed `H4_norm` configuration
  lock.
- Three 50-epoch locked training runs.
- Single authorized train/validation/test evaluation per seed.
- Slice-micro, patient-macro, patient-cluster, field-strength-stratified, NLL,
  Brier, ECE, and reliability analysis.
- Public, de-identified tables and professional figures.

## Locked seed summary

| Seed | Epochs | Best epoch | Best validation accuracy | Test accuracy |
|---:|---:|---:|---:|---:|
| 2025 | 50/50 | 33 | 0.9871 | 0.5245 |
| 2026 | 50/50 | 9 | 0.9897 | 0.5140 |
| 2027 | 50/50 | 11 | 0.9911 | 0.5141 |

## Reproducibility and privacy state

- Official archive MD5: `dbe61da7dc7b06c69c10dbbea0a13b40`.
- Audited manifest SHA-256:
  `44b4fa5da873efabc323d0871037da96de494c38020d41c499c644c149d6561e`.
- Model/checkpoint selection used validation accuracy, lower validation NLL as
  the tie-breaker, then the earlier epoch.
- Threshold, preprocessing, seeds, and inclusion rules were not altered after
  test access.
- Public result outputs were scanned for patient IDs, image paths, and dataset
  filenames; the audit was clean.

## Remaining optional work

The core reproduction is finished. Remaining work is optional extension work:

1. ERM versus inverse-group weighting or GroupDRO on a new patient-disjoint
   extension split with group metrics and calibration.
2. Anatomy-preserving preprocessing robustness.

These extensions require new experiments and must not be presented as part of
the locked Table 2 reproduction. The GroupDRO comparison is the only remaining
substantially GPU-expensive stage.

See `reports/FINAL_REPORT.md` for the interpretation and
`reports/final_results/` for the public evidence.
