# Paper summary

## Citation

Won JB, De Neve W, Vankerschaver J, Ozbulak U. *SpurBreast: A Curated
Dataset for Investigating Spurious Correlations in Real-world Breast MRI
Classification.* MICCAI 2025. DOI: 10.1007/978-3-032-05325-1_53.

## Problem

The paper introduces controlled breast-MRI datasets in which non-clinical
features are deliberately correlated with slice labels. The goal is to expose
shortcut learning: high validation performance caused by a spurious feature
that fails when the relationship is broken at test time.

## Clinical task

All upstream patients have biopsy-confirmed invasive breast cancer. Individual
2D slices are labelled tumor-containing or non-tumor. Tumor-negative slices do
not represent cancer-negative patients. A buffer zone around the transition
between tumor and non-tumor slices is excluded.

## Target experiment

The approved target is the ResNet-50 magnetic-field-strength row in Table 2.
Training and validation tumor slices are taken only from 1.5 T examinations;
non-tumor slices are taken only from different 3 T patients. The fixed test set
contains both labels for each patient.

Published results:

| Split | Accuracy | PPV | NPV |
|---|---:|---:|---:|
| Training | 0.99 | 0.98 | 0.99 |
| Validation | 0.99 | 1.00 | 0.98 |
| Test | 0.52 | 0.62 | 0.41 |

## Method and optimization evidence

- PyTorch ResNet-50 initialized from ImageNet weights.
- Full training grid: SGD, Adam, AdamW; learning rates `1e-5`, `1e-4`,
  `1e-3`, `1e-2`; weight decay `1e-4`, `1e-5`, or zero.
- Cosine annealing for SGD.
- Batch size 32 and 50 epochs.
- Highest validation-accuracy checkpoint selected.

The winning configuration, loss, normalization, exact weights, seeds, metric
aggregation, and threshold are not reported. No supplementary material was
submitted and no complete training implementation was released.
