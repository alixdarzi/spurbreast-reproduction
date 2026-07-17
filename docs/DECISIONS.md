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

## 2026-07-13 — Post-Gate-3 execution controls

The approved sensitivity screen uses frozen inherited configurations H1–H4
for 10 epochs and seed 2025. A provenance-matched wrapper starts, resumes, or
skips each run after verifying the archive MD5 and audited manifest hash.
Selection reads validation summaries only. F1–F2 run only below the declared
0.97 validation-accuracy trigger; normalization runs only for the winning
optimization. Three expanded 50-epoch seed configs are generated only after
the selector reaches `ready_to_lock`, and must be committed before test access.

## 2026-07-13 — Colab Python 3.12 compatibility

The hosted Colab T4 runtime presented Python 3.12, while the initial package
metadata declared support only through Python 3.11. The already-pinned shared
dependencies support Python 3.12, so `requires-python` was widened from
`>=3.10,<3.12` to `>=3.10,<3.13`; no scientific configuration or dependency
version changed. The persistent notebook now requires a clean tracked worktree
and performs a fast-forward-only pull from `origin/main` on setup so resumed
sessions cannot silently run stale source code.

## 2026-07-13 — Validation-only configuration lock

All four primary sensitivity runs completed without test access. H4 (SGD,
learning rate 0.01, weight decay 1e-4, cosine schedule) was the optimization
winner with validation accuracy 0.9894 and NLL 0.0319. Its prespecified
ImageNet-normalized counterpart reached 0.9916 and 0.0279, so the frozen
`final_winner` is `H4_norm`. The 0.97 fallback threshold was exceeded and F1–F2
were not run. Locked configs use 50 epochs and seeds 2025, 2026, and 2027;
`LOCK.json` records `test_status: not_evaluated`.

## 2026-07-16 — Colab ephemeral local data cache

The first seed-2026 attempt produced no history or checkpoint because random
reads from mounted Drive left the data-loader workers blocked. It was stopped
before any completed epoch. The official archive was copied to ephemeral Colab
local disk, its published MD5 was reverified, and exactly 19,926 PNGs were
extracted. The unchanged extracted tree is bind-mounted over the configured
project-relative `data/raw/field_strength` directory during training.

This is an operational I/O optimization only: no config, manifest, image byte,
cohort, split, sample order, transform, model, optimizer, seed, checkpoint rule,
or metric changed. Results, checkpoints, and registry records remain
Drive-persistent. The cache and bind mount must be rebuilt after a runtime
reset. The test split remains unevaluated.

## 2026-07-16 — Free Colab disconnect during seed 2027

Locked seed 2026 completed all 50 epochs and the continuation cell started seed
2027 automatically. Seed 2027's persisted history was directly verified at 20
completed epochs before the hosted runtime later disconnected. Immediate GPU
reconnect attempts returned `Unable to connect to the runtime`.

The conservative resume boundary is therefore seed 2027 epoch 20/50, although
the epoch-boundary checkpoint on Drive may be newer because training continued
after the last direct check. On reconnect, audit the persisted history and
checkpoint metadata first, rebuild the checksum-verified ephemeral data cache,
and let `scripts/run_or_resume.py` select the newest provenance-matched
checkpoint. Do not restart from an earlier epoch manually, change the locked
configuration, or access test results. Seed 2026 must be skipped as completed.

## 2026-07-17 — Trusted checkpoint loading on newer PyTorch

The resumed Colab runtime supplied PyTorch 2.11, whose default restricted
checkpoint loader rejected the NumPy RNG state saved for exact epoch-boundary
resume. The project now loads only its own provenance-checked training and
evaluation checkpoints with `weights_only=False` and stages all checkpoint
tensors on CPU. CPU staging is required because DataLoader and RNG generator
states must remain CPU byte tensors; model and optimizer loaders perform their
normal device transfer afterward. This explicitly preserves the historical
behavior required for optimizer, scaler, DataLoader generator, and RNG
restoration. It does not change model weights, training configuration, data,
splits, metrics, or checkpoint selection. A regression test covers NumPy RNG
and DataLoader generator state.
