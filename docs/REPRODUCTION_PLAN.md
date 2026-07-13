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

## Primary metrics

Accuracy, PPV, and NPV are computed from a single global slice-level confusion
matrix. Patient-macro metrics and patient-cluster bootstrap intervals are
reported separately. Test thresholds are never optimized.

## Scientific shortcut analysis

In addition to the Table 2 metrics, report predicted-positive rates and
classification metrics stratified by 1.5 T versus 3 T. A credible reproduction
must demonstrate the correct field-strength shortcut direction and a large
validation-to-test performance gap.

## Execution

- Local GTX 1660 Ti: audit, tests, reduced smoke run, resume test.
- Free Colab GPU with at least 12 GB: locked batch-32 FP32 final runs.
- Save atomic `latest` and `best` checkpoints every epoch, including RNG state.
- Stop before full training at Approval Gate 3.
