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

## 2026-07-13 — Released-archive audit

The checksum-verified archive contains 19,926 grayscale 320×320 PNGs from 700
unique patients. Full decoding and pixel hashing found no corrupt images, no
duplicate image hashes, no cross-split duplicate hashes, and no cross-split
patient overlap. The released folder membership is therefore the immutable
primary split for this close reproduction.

## 2026-07-13 — Shortcut-oracle discrepancy

The field-strength oracle is perfect on the released training and validation
sets, confirming the intended shortcut construction. On the released test set
it produces TP=1,394, FP=1,684, TN=1,710, and FN=2,000 (accuracy 0.4573, PPV
0.4529, NPV 0.4609). These values are reported as an audit result, not adjusted
to resemble Table 2. Model and hyperparameter choices remain validation-only;
the test split will not be used to resolve this discrepancy.

## 2026-07-13 — Visual preprocessing audit

The primary transform remains the official repository example, including the
default `RandomResizedCrop(224)` after resize to 256. A seed-2025 montage showed
that one sampled crop removed the visible enhancing lesion from a positive
slice. This is documented as an augmentation risk; it does not justify changing
the close-reproduction transform after viewing test data.
