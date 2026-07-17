# Manageable post-reproduction extensions

The primary reproduction is complete and locked. These are optional follow-on
experiments; neither changes or replaces the reported Table 2 result.

## 1. ERM versus group reweighting or GroupDRO

The primary training split contains only two of four label × field groups, so
GroupDRO is not identifiable there. After primary evaluation, create a new
patient-disjoint 90/30/30 extension split from the 150 released test patients,
where both labels occur for each patient. Compare ERM, inverse-group weighting,
and GroupDRO with identical ResNet-50 compute. Report pooled, patient-macro,
1.5 T/3 T, NLL, Brier, ECE, and reliability results. Fit temperature scaling
on extension validation only. This is a new mitigation experiment, not a
replacement Table 2 result.

## 2. Anatomy-preserving preprocessing robustness

Compare the official random-resized crop with fixed resize and one mild
anatomy-preserving crop/affine pipeline. Select on validation, lock variants,
then evaluate once. Report performance, field-stratified positive rates, and
calibration. This directly tests the lesion-excluding crop observed in the
visual audit without adding another architecture.

Together these add robust optimization, group-aware metrics, calibration, and
augmentation analysis around one focused question. Avoiding segmentation,
multimodal fusion, and deployment infrastructure keeps the portfolio deep but
manageable.
