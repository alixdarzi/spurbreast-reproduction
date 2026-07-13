# Released data audit

Audit date: 2026-07-13. The source is the official Zenodo
`field_strength.zip` release, verified before extraction.

## Integrity and provenance

| Item | Verified value |
|---|---|
| Archive bytes | 905,891,386 |
| Archive MD5 | `dbe61da7dc7b06c69c10dbbea0a13b40` |
| Archive SHA-256 | `c6c4b6bcc62168cc91c7e44bc0e5e19bff75a1232c7284bc682e0ccb6ce9bafc` |
| Internal manifest SHA-256 | `44b4fa5da873efabc323d0871037da96de494c38020d41c499c644c149d6561e` |
| Images decoded | 19,926 of 19,926 |
| Image format | 320×320, grayscale (`L`) |
| Corrupt images | 0 |
| Duplicate pixel-hash groups | 0 |
| Cross-split duplicate-hash groups | 0 |
| Cross-split patient overlaps | 0 |

## Released split membership

| Split/label | Images | Patients | Field strength |
|---|---:|---:|---|
| Training/non-tumor | 4,781 | 200 | all 3 T |
| Training/tumor | 4,781 | 200 | all 1.5 T |
| Validation/non-tumor | 1,788 | 75 | all 3 T |
| Validation/tumor | 1,788 | 75 | all 1.5 T |
| Test/non-tumor | 3,394 | 150 | 1,684 at 1.5 T; 1,710 at 3 T |
| Test/tumor | 3,394 | 150 | 1,394 at 1.5 T; 2,000 at 3 T |

Training has 400 unique patients, validation 150, and test 150. Training and
validation use disjoint patients between labels; every test patient contributes
both tumor and non-tumor slices. The test cohort contains 71 patients at 1.5 T
and 79 at 3 T.

## Deterministic shortcut baselines

The paper describes the learned shortcut direction as tumor for 1.5 T and
non-tumor for 3 T. Applying that rule directly gives:

| Split | Accuracy | PPV | NPV |
|---|---:|---:|---:|
| Training | 1.000 | 1.000 | 1.000 |
| Validation | 1.000 | 1.000 | 1.000 |
| Test | 0.457 | 0.453 | 0.461 |

The test confusion matrix is TP=1,394, FP=1,684, TN=1,710, FN=2,000. This is
not a learned-model result and is not used for model selection. It confirms
the intended training/validation shortcut but conflicts with both the paper's
qualitative test narrative and its reported 0.52/0.62/0.41 test metrics under
standard pooled slice definitions. The discrepancy remains explicit rather
than being resolved through test-set tuning.

Machine-readable evidence is stored in
`reports/tables/data_audit_summary.json` and
`reports/tables/shortcut_baselines.json`.

## Preprocessing visual check

The repository-matched pipeline converts each grayscale PNG to three-channel
RGB. Training resizes to 256×256 and then applies torchvision's
`RandomResizedCrop(224)` defaults before tensor conversion; evaluation directly
resizes to 224×224. Neither path normalizes intensities in the primary setup.

A deterministic local montage (seed 2025) confirmed the expected geometry and
channel handling. In the sampled positive slice, the random crop excluded the
visibly enhanced lesion while the evaluation resize retained it. This is a
plausible consequence of the official example transform, not evidence of a
loader error. The primary reproduction keeps it for traceability; the bounded
validation-only normalization/preprocessing sensitivity is retained as a risk
check. The montage is ignored by Git because it is a medical-image derivative.
