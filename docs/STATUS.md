# Project checkpoint and resume status

Last updated: 2026-07-14 (Asia/Tehran)

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
`test_status: not_evaluated`. Locked seed 2025 completed all 50 epochs and its
best checkpoint is safely persisted on Drive. Seeds 2026 and 2027 have not
started. No learned checkpoint has been evaluated on the test split, so no
Table 2 reproduction result is claimed yet.

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

## Locked-seed progress

| Seed | Epochs | Status | Best epoch | Best val accuracy | Best val NLL | Runtime |
|---:|---:|---|---:|---:|---:|---:|
| 2025 | 50/50 | Completed | 33 | 0.9871 | 0.0423 | 7,488 s (2 h 5 min) |
| 2026 | 0/50 | Not started | — | — | — | — |
| 2027 | 0/50 | Not started | — | — | — | — |

Seed 2025 used the committed `H4_norm` lock and completed with return code 0.
Its result directory is
`locked_table2_field_strength_resnet50-seed2025-20260713T175147Z`.
The test split remains unseen.

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

Seed 2025 completed before the hosted runtime disconnected. The queued
continuation cell did not execute after the disconnection, so seeds 2026 and
2027 were not created. A subsequent reconnect attempt on 2026-07-14 was denied
because the free Colab account had reached its temporary GPU usage limit. A CPU
runtime was used only to audit the persisted Drive summaries; no training or
test evaluation was run on that CPU runtime.

The current safe boundary is after locked seed 2025. The final-evaluation cell
still has `ALLOW_FINAL_TEST = False` and has never run.

## Exact next execution procedure

1. Wait for free Colab GPU access to become available again; do not run the
   locked training on the temporary CPU runtime.
2. Reconnect to a T4, mount Drive, and verify the repository is still at the
   published lock commit with a clean tracked worktree.
3. Run seed 2026 from `configs/locked/seed2026.yaml`, then seed 2027 from
   `configs/locked/seed2027.yaml`. The resume wrapper makes either run safe to
   restart if Colab disconnects again.
4. Do not inspect test metrics during either seed run.
5. Confirm all three best checkpoints and completed summaries exist.
6. Perform the single authorized evaluation of train, validation, and test for
   each locked seed. Do not use test outcomes to change preprocessing, training,
   checkpoint selection, thresholds, or seed inclusion.
7. Aggregate global, patient-macro, and field-strength-stratified metrics,
   calibration, confidence intervals, and across-seed variability.

## Remaining work

- Wait for free Colab GPU quota to recover.
- Run locked seeds 2026 and 2027 for 50 epochs each.
- Perform the single locked train/validation/test evaluation.
- Aggregate global and field-strength-stratified metrics, calibration,
  uncertainty intervals, and seed variability.
- Run the approved ERM versus group reweighting or GroupDRO extension and one
  additional manageable extension after the primary reproduction is secure.
- Complete the final report, figures, limitations, and portfolio presentation.

Based on seed 2025's measured 2-hour-5-minute runtime, seeds 2026 and 2027
should require roughly 4.2 additional GPU-hours in total. Budget 5–7 remaining
Colab GPU-hours including evaluation and operational overhead; this excludes
quota recovery time, service queueing, and interrupted free runtimes.
