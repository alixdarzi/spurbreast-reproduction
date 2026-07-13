# Project checkpoint and resume status

Last updated: 2026-07-13 (Asia/Tehran)

## Current state

The evidence review, reproduction protocol, repository implementation, released
archive audit, preprocessing inspection, regression tests, and real-data CPU
checkpoint/resume smoke test are complete. The public repository is available
at <https://github.com/alixdarzi/spurbreast-reproduction>.

GPU sensitivity training has **not** started. No learned checkpoint has been
evaluated on the test split, and no Table 2 reproduction result is claimed yet.
This is the safe boundary at which work was paused.

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

## Colab checkpoint

Persistent notebook copy:
<https://colab.research.google.com/drive/1rv8ogoWYrrsLDxdHU2pyOUzoKPGPZgoR>

The Drive-owned copy exists and was connected to a hosted T4 GPU, but the live
runtime repeatedly returned to **Connecting**. Earlier attempts also produced
`[object CloseEvent]` and one `ValueError: mount failed`. Cell 1 did not reach
`Mounted at /content/drive`, so the repository was not cloned to Drive and no
download, package installation, GPU audit, or training began in Colab.

The repeated loss of a freshly connected T4 is consistent with the VPN or its
route interrupting Colab's long-lived browser/runtime connection. The Drive
mount error may still be a separate Google Drive authorization or Drive-layout
problem and must be checked after the connection is stable.

## Exact resume procedure after changing the VPN

1. Open the persistent notebook link above and reload it.
2. Connect to a hosted GPU and confirm the status stays connected and identifies
   a T4 for several minutes.
3. Run Cell 1 and approve Google Drive access. Do not continue unless its output
   confirms `Mounted at /content/drive` and shows the Drive project path.
4. Run Cell 2. Confirm CUDA is available, record the actual Torch/CUDA/GPU
   versions, and retain batch size 32 unless a protocol deviation is documented.
5. Run Cell 3. It performs the checksum-verified download, deterministic data
   preparation/audit, and regression tests. Confirm all integrity checks pass.
6. Begin validation-only screening with H1. Run H2, H3, and H4 in later sessions,
   using the resume wrapper. Do not run final-evaluation cells.
7. Follow the selector's requested fallback/normalization run. Write and commit
   the lock only when its status is `ready_to_lock`.
8. Run all three locked 50-epoch seeds. Test access remains one-time and only
   after the committed lock and all prespecified seeds are complete.

If Cell 1 fails again with the VPN disabled, use Colab's **Disconnect and delete
runtime**, reconnect once, and retry authorization. If the fresh runtime still
cannot mount Drive, inspect the Drive-specific cause before starting any
ephemeral `/content` training; checkpoints and run records must remain
persistent and resumable.

## Remaining work

- Establish a stable Colab T4 plus persistent Drive mount.
- Complete Cell 2 and Cell 3 environment/data verification.
- Run H1–H4 and any prespecified selector-requested sensitivity runs.
- Freeze, commit, and publish the selected configuration lock.
- Run seeds 2025, 2026, and 2027 for 50 epochs each.
- Perform the single locked train/validation/test evaluation.
- Aggregate global and field-strength-stratified metrics, calibration,
  uncertainty intervals, and seed variability.
- Run the approved ERM versus group reweighting or GroupDRO extension and one
  additional manageable extension after the primary reproduction is secure.
- Complete the final report, figures, limitations, and portfolio presentation.

Estimated remaining Colab GPU time is approximately 8–16 hours, spread across
resumable sessions. This estimate excludes service queueing, interrupted free
runtimes, and the one-time data transfer/audit.
