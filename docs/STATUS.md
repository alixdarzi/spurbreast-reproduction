# Project checkpoint and resume status

Last updated: 2026-07-13 (Asia/Tehran)

## Current state

The evidence review, reproduction protocol, repository implementation, released
archive audit, preprocessing inspection, regression tests, and real-data CPU
checkpoint/resume smoke test are complete. The public repository is available
at <https://github.com/alixdarzi/spurbreast-reproduction>.

The complete validation-only sensitivity screen has finished on a Colab T4.
H4 (SGD, learning rate 0.01, cosine schedule) won the optimizer screen, and its
prespecified normalized counterpart improved both validation accuracy and NLL.
The final winner is therefore `H4_norm`. The three 50-epoch seed configs and
their hashes are written under `configs/locked/`, with
`test_status: not_evaluated`. No locked seed has started and no learned
checkpoint has been evaluated on the test split, so no Table 2 reproduction
result is claimed yet.

## Verified scientific and data facts

- Target: Table 2, Magnetic Field Strength — ResNet-50.
- Task: tumor-containing versus non-tumor 2D MRI slice classification among
  patients with biopsy-confirmed invasive breast cancer; not patient-level
  cancer diagnosis.
- Official archive: `field_strength.zip`, 905,891,386 bytes.
- Verified MD5: `dbe61da7dc7b06c69c10dbbea0a13b40`.
- Verified SHA-256:
  `c6c4b6bcc62168cc91c7e44bc0e5e19bff75a1232c7284bc682e0ccb6ce9bafc`.
- Released data: 19,926 valid 320×320 grayscale PNGs from 700 patients.
- Split counts: training 9,562 images/400 patients; validation 3,576/150;
  test 6,788/150.
- Cross-split patient overlap, corrupt images, duplicate image hashes, and
  cross-split duplicate hashes: zero.
- Training and validation have the intended perfect field-strength shortcut;
  every test patient has both slice labels.
- Audited manifest SHA-256:
  `44b4fa5da873efabc323d0871037da96de494c38020d41c499c644c149d6561e`.
- The released-split field-strength oracle has test accuracy 0.4573, PPV
  0.4529, and NPV 0.4609. This audit discrepancy will not be used for tuning.
- The published balanced-test values 0.52/0.62/0.41 cannot all come from one
  standard globally pooled confusion matrix. Correct global metrics will be
  retained, with patient-macro metrics reported separately.

## Validation-only sensitivity result

| ID | Optimizer / preprocessing | Best epoch | Val accuracy | Val NLL |
|---|---|---:|---:|---:|
| H1 | AdamW, 1e-4, raw input | 4 | 0.9785 | 0.0462 |
| H2 | AdamW, 1e-3, raw input | 9 | 0.9606 | 0.1048 |
| H3 | Adam, 1e-4, raw input | 4 | 0.9785 | 0.0462 |
| H4 | SGD, 0.01, cosine, raw input | 7 | 0.9894 | 0.0319 |
| H4_norm | H4 with ImageNet normalization | 7 | **0.9916** | **0.0279** |

Selection used validation accuracy, then lower NLL, then the earlier epoch.
The selector status is `ready_to_lock`, fallback runs were not triggered, and
the final winner is `H4_norm`. These results are recorded in
`reports/tables/sensitivity_selection.json`; the test split was not loaded.

## Completed engineering safeguards

- Frozen H1–H4 validation-only sensitivity configurations.
- Resumable runs with provenance and archive/manifest verification.
- Atomic best/latest checkpoints with optimizer, scaler, random-number, and
  data-loader state.
- Validation-only selection and a mandatory committed configuration lock before
  final seeds or any test evaluation.
- Patient-disjoint split assertions, deterministic ordering, strict metrics,
  preprocessing visualization, and tensor-range checks.
- CPU real-image smoke test paused after epoch 0 and resumed through epoch 1;
  it constructed no test DataLoader.
- Regression suite and notebook validation passed before publication.
- Five validation-only sensitivity runs completed with Drive-persistent
  checkpoints and registry records; peak observed VRAM stayed below 3.3 GB.
- The final lock selects `H4_norm` for seeds 2025, 2026, and 2027, keeps the
  physical batch size at 32, and records `test_status: not_evaluated`.

## Colab checkpoint

Persistent notebook copy:
<https://colab.research.google.com/drive/1rv8ogoWYrrsLDxdHU2pyOUzoKPGPZgoR>

Changing the VPN route resolved the connection problem. The notebook mounted
Drive, cloned the repository, verified a Tesla T4 with 14.6 GiB VRAM, installed
the Python 3.12-compatible project, downloaded and audited the official archive,
and passed all regression tests. H1, H2, H3, H4, and H4_norm then completed with
checkpoints and results persisted under Google Drive.

The current safe boundary is after validation selection and lock generation.
The final-evaluation cell still has `ALLOW_FINAL_TEST = False` and has never run.

## Exact next execution procedure

1. Commit and push `configs/locked/`, the sensitivity selection report, this
   status update, and the sensitivity registry rows.
2. In Colab, fast-forward the Drive clone to that lock commit and verify a clean
   worktree. Do not regenerate or alter the lock after this point.
3. Run locked seed 2025 for 50 epochs with the resume wrapper.
4. Run seeds 2026 and 2027 in later resumable sessions using their committed
   configs. Do not inspect test metrics during any seed run.
5. Confirm all three best checkpoints and completed summaries exist.
6. Perform the single authorized evaluation of train, validation, and test for
   each locked seed. Do not use test outcomes to change preprocessing, training,
   checkpoint selection, thresholds, or seed inclusion.
7. Aggregate global, patient-macro, and field-strength-stratified metrics,
   calibration, confidence intervals, and across-seed variability.

## Remaining work

- Commit and publish the completed `H4_norm` configuration lock.
- Run seeds 2025, 2026, and 2027 for 50 epochs each.
- Perform the single locked train/validation/test evaluation.
- Aggregate global and field-strength-stratified metrics, calibration,
  uncertainty intervals, and seed variability.
- Run the approved ERM versus group reweighting or GroupDRO extension and one
  additional manageable extension after the primary reproduction is secure.
- Complete the final report, figures, limitations, and portfolio presentation.

Based on the measured 25-minute, 10-epoch sensitivity runs, the three 50-epoch
locked seeds should require roughly 6.3 GPU-hours in total, plus evaluation and
report generation. Budget 7–9 Colab GPU-hours across resumable sessions; this
excludes service queueing and interrupted free runtimes.
