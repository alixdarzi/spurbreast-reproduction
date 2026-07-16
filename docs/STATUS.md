# Project checkpoint and resume status

Last updated: 2026-07-16 (Asia/Tehran)

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
`test_status: not_evaluated`. Locked seeds 2025 and 2026 completed all 50
epochs, and their checkpoints are safely persisted on Drive. Seed 2027 started
automatically; 20 completed epochs were directly verified before the free
Colab runtime disconnected. The persisted latest checkpoint may be newer and
will be audited on reconnect. No learned checkpoint has been evaluated on the
test split, so no Table 2 reproduction result is claimed yet.

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
| 2026 | 50/50 | Completed | Pending summary audit | — | — | About 1 h 48 min |
| 2027 | 20/50+ verified | Runtime disconnected | Pending resume audit | — | — | Partial |

Seed 2025 used the committed `H4_norm` lock and completed with return code 0.
Its result directory is
`locked_table2_field_strength_resnet50-seed2025-20260713T175147Z`. Seed 2026
completed in
`locked_table2_field_strength_resnet50-seed2026-20260716T135531Z`; its
`history.jsonl` was directly verified at 50 records with final zero-based epoch
index 49. Seed 2027 is in
`locked_table2_field_strength_resnet50-seed2027-20260716T154311Z`; its history
was directly verified at 20 records with final zero-based epoch index 19. The
test split remains unseen.

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

Seed 2025 completed before the hosted runtime disconnected. The free-tier GPU
quota subsequently recovered, and a fresh Tesla T4 runtime was connected on
2026-07-16. The Drive clone was verified at the published lock commit with a
clean tracked worktree. The first seed-2026 attempt stalled before producing an
epoch because mounted-Drive random reads blocked its data-loader workers; it
was stopped without discarding any valid completed epoch.

For the replacement run, the checksum-verified official archive was copied to
ephemeral Colab local disk, its MD5 was reverified, and all 19,926 PNGs were
extracted. That unchanged extracted tree is bind-mounted over the configured
project-relative `data/raw/field_strength` path for faster reads. Results,
checkpoints, and registry records still persist to Drive. This runtime-only I/O
change does not alter the archive, manifests, split, sample order, transforms,
model, optimizer, or seeds. It must be recreated after a runtime reset.

Seed 2026 completed all 50 epochs normally. The same notebook cell then started
seed 2027 automatically. Seed 2027 reached at least 20 completed epochs; the
hosted runtime remained active after that direct check, so a newer persisted
checkpoint may exist. Later that evening the runtime disconnected. Two
reconnect attempts returned `Unable to connect to the runtime`, consistent
with a temporary free-tier availability or usage limit.

The notebook reports `All changes saved`. Results and epoch-boundary latest/best
checkpoints were written to Drive, so the resume wrapper can safely inspect and
continue the newest compatible seed-2027 run. The conservative known-safe
boundary is 20/50; do not assume later epochs until the Drive files are audited
after reconnect. The final-evaluation cell still has
`ALLOW_FINAL_TEST = False` and has never run.

## Exact next execution procedure

1. Retry Colab later and require a GPU runtime; do not continue this locked run
   on CPU.
2. Mount Drive, run the setup cell, and verify the repository/config lock. A
   fast-forward pull containing documentation-only updates is acceptable.
3. Inspect the persisted seed-2027 `history.jsonl`, `latest.pt`, `best.pt`, and
   summary before launching anything. Treat 20/50 as the conservative boundary
   until this audit establishes a newer completed epoch.
4. Rebuild the ephemeral local data cache from the existing official archive,
   reverify MD5 `dbe61da7dc7b06c69c10dbbea0a13b40`, confirm 19,926 PNGs,
   and restore the bind mount over project-relative
   `data/raw/field_strength`.
5. Rerun the locked-seed continuation cell. The wrapper should skip completed
   seed 2026 and resume seed 2027 from its newest provenance-matched checkpoint.
6. Do not inspect or evaluate the test split while seed 2027 is incomplete.
7. Confirm all three best checkpoints and completed summaries exist.
8. Perform the single authorized evaluation of train, validation, and test for
   each locked seed. Do not use test outcomes to change preprocessing, training,
   checkpoint selection, thresholds, or seed inclusion.
9. Aggregate global, patient-macro, and field-strength-stratified metrics,
   calibration, confidence intervals, and across-seed variability.

## Remaining work

- Reconnect to a Colab GPU and audit the newest persisted seed-2027 checkpoint.
- Resume and finish locked seed 2027; at least 20/50 epochs are verified.
- Perform the single locked train/validation/test evaluation.
- Aggregate global and field-strength-stratified metrics, calibration,
  uncertainty intervals, and seed variability.
- Run the approved ERM versus group reweighting or GroupDRO extension and one
  additional manageable extension after the primary reproduction is secure.
- Complete the final report, figures, limitations, and portfolio presentation.

Seed 2026 completed in roughly 1 hour 48 minutes with local data caching. From
the conservative seed-2027 boundary of 20/50, at most about 65 minutes of
locked training remain at the observed rate; a newer persisted checkpoint may
reduce this. Budget 2–4 available Colab GPU-hours for resume validation, the
remaining training, locked evaluation, and operational overhead. This excludes
free-tier quota recovery and service queueing.
