# Decision log

## 2026-07-13 — Approved target

- Target: Table 2, Magnetic Field Strength — ResNet-50.
- Category: close reproduction.
- Primary dataset: official `field_strength.zip` only.
- Primary scientific focus: patient-disjoint field-strength shortcut protocol.

## 2026-07-13 — Primary preprocessing

Use the official repository example as the highest available implementation
evidence: RGB, resize to 256, random resized crop to 224 for training; resize
to 224 for evaluation; `ToTensor`; no normalization. ImageNet normalization is
a separately labelled sensitivity analysis.

## 2026-07-13 — Missing method details

Use `IMAGENET1K_V1`, full fine-tuning, unweighted cross-entropy, two-logit
argmax, and PyTorch optimizer defaults unless a parameter appears in the
paper. These are assumptions, not claims about the authors' implementation.

## 2026-07-13 — Metric integrity

Use one globally pooled slice confusion matrix for primary accuracy, PPV, and
NPV. The published balanced-test values 0.52/0.62/0.41 are mathematically
incompatible under this definition. Do not change correct metrics to force a
match; report patient-macro metrics separately.

## 2026-07-13 — Engineering corrections

Use sorted file paths and `pathlib` rather than reproducing the official
loader's unsorted `os.listdir` order and forward-slash filename parsing. These
changes improve determinism and Windows compatibility without changing cohort
membership or image content.

## 2026-07-13 — Download reliability

The project downloader uses checksum-verified, resumable 16 MiB byte ranges
with one worker by default. Eight parallel Zenodo requests stalled on the local
connection, so concurrency is opt-in. For the initial local audit, Windows BITS
was used as an OS-managed persistent transfer; the result is accepted only
after matching the same published MD5.

## 2026-07-13 — Exact epoch-boundary resume

DataLoader workers are not persisted between epochs. Their seeds are derived
from a saved generator state, so resuming after an epoch does not depend on
uncaptured persistent-worker augmentation state. The small worker startup cost
is accepted for stronger provenance.
