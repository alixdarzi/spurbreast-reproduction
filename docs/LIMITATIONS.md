# Known limitations

- No supplementary material was submitted.
- The official repository has no training loop, evaluation code, configuration,
  requirements, split-generation code, or checkpoints.
- The winning optimizer, learning rate, weight decay, and epoch are unknown.
- Loss, class sampling, normalization, exact ImageNet weight version, precision,
  optimizer defaults, seeds, threshold, and metric aggregation are unknown.
- The DICOM sequence/contrast phase, intensity conversion, and buffer width
  used to construct PNGs are unavailable.
- The paper reports ten randomized patient-split repetitions, but only one
  split is published. Three training seeds on that split are not equivalent.
- Test accuracy, PPV, and NPV in Table 2 cannot all arise from one standard
  confusion matrix on the released balanced test set.
- The exact documented shortcut rule (predict tumor for every 1.5 T slice and
  non-tumor for every 3 T slice) gives 0.457 accuracy, 0.453 PPV, and 0.461 NPV
  on the released test folders. It therefore cannot by itself explain the
  reported 0.52/0.62/0.41 test row. A different metric aggregation, different
  split realization, or an imprecise qualitative description may be involved.
- Slice-level samples within a patient are correlated; naive per-slice
  confidence intervals would be invalid.
- Free Colab hardware and session duration are not guaranteed.
- The local GTX 1660 Ti has 6 GB VRAM and may not fit batch-32 FP32 training.
- Data are CC BY-NC 4.0. Model-weight redistribution remains to be reviewed.
- The final three-seed experiment reproduces test accuracy but not the paper's
  test PPV/NPV; this is reported as an unresolved metric-provenance discrepancy.
- Test ECE is descriptive only. Temperature scaling or any other calibration
  fit on test would be invalid and was not performed.
