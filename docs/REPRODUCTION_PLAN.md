# Reproduction plan

## Category

**Close reproduction:** same official curated images, same released split,
same ResNet-50 family and ImageNet initialization, and the same target table
row. Missing training and metric details are resolved through documented
assumptions and a bounded validation-only sensitivity study.

## Primary protocol

1. Verify the official archive checksum and extract it without changing files.
2. Join filename patient identifiers to the official TCIA field-strength
   metadata.
3. Assert published counts, patient disjointness, label direction, and unbiased
   test construction.
4. Load PNGs through PIL and convert to RGB.
5. Training preprocessing: resize 256×256, random resized crop 224×224,
   tensor conversion, no normalization.
6. Evaluation preprocessing: resize 224×224 and tensor conversion.
7. Fine-tune all ResNet-50 parameters from `IMAGENET1K_V1` weights.
8. Use unweighted cross-entropy, batch size 32, and 50 epochs.
9. Choose optimization only from the prespecified validation-only sensitivity
   plan in `configs/sensitivity.yaml`.
10. Select maximum validation accuracy, then lower validation NLL, then the
    earliest epoch.
11. Lock the configuration before evaluating the test split.
12. Report seeds 2025, 2026, and 2027 without seed selection.

The full archive audit has completed successfully. The fixed manifest SHA-256
is `44b4fa5da873efabc323d0871037da96de494c38020d41c499c644c149d6561e`.
See `docs/DATA_AUDIT.md` for verified counts and shortcut-oracle results.

## Primary metrics

Accuracy, PPV, and NPV are computed from a single global slice-level confusion
matrix. Patient-macro metrics and patient-cluster bootstrap intervals are
reported separately. Test thresholds are never optimized.

## Scientific shortcut analysis

In addition to the Table 2 metrics, report predicted-positive rates and
classification metrics stratified by 1.5 T versus 3 T. A credible reproduction
must demonstrate the correct field-strength shortcut direction and a large
validation-to-test performance gap.

The non-learned field oracle is an audit baseline, not a selectable model. It
is perfect on training/validation and reaches only 0.457 slice accuracy on the
released test set; this is evidence of shortcut failure and also exposes an
unresolved disagreement with the published test metrics.

## Execution

- Local GTX 1660 Ti: audit, tests, reduced smoke run, resume test.
- Free Colab GPU with at least 12 GB: locked batch-32 FP32 final runs.
- Save atomic `latest` and `best` checkpoints every epoch, including RNG state.
- Verify resume by intentionally pausing the two-epoch smoke run after epoch 0
  and completing epoch 1 from `latest.pt`.
- Stop before full training at Approval Gate 3.

## Approval Gate 3 execution order

1. Screen H1–H4 for 10 epochs each using training and validation only.
2. Run the normalization sensitivity for 10 epochs only with the winning
   optimization setup. Run F1–F2 only if no primary setup reaches 0.97
   validation accuracy.
3. Commit the locked preprocessing, optimizer, learning rate, weight decay,
   and checkpoint rule before any learned-model test evaluation.
4. Train the locked 50-epoch setup for seeds 2025, 2026, and 2027.
5. Evaluate every locked seed once; never select seeds or thresholds on test.
