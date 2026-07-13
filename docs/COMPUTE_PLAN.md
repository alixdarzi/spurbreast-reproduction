# Compute and storage plan

These are conservative planning estimates. The first GPU epoch must record
actual wall time and `torch.cuda.max_memory_allocated()`.

## Measured storage

| Artifact | Size |
|---|---:|
| Official ZIP | 0.906 GB |
| Extracted 19,926 PNGs | 0.902 GB |
| One best/latest checkpoint | 0.283 GB |
| Free C: space at Gate 3 | 441 GB |

Budget 6–8 GB for one complete working copy, or 10–12 GB on persistent Drive
when retaining sensitivity and seed checkpoints.

## GTX 1660 Ti (6 GB)

- Batch-32 FP32 peak estimate: 7–10 GB; likely out of memory.
- Fallback batch 8 with AMP and four-step accumulation: 3.5–5.5 GB estimated.
  This changes physical batch and precision, so it is not the primary run.
- One 50-epoch run: about 3–6 hours; full bounded plan: 15–28 hours.
- The verified local smoke environment is CPU-only; CUDA 11.7 requirements
  are supplied but were deliberately not installed for the audit.

## Free Google Colab

- Target T4-class 16 GB GPU or better; batch-32 FP32 peak estimate 8–12 GB.
- One 50-epoch run: about 1.5–3 hours.
- Four 10-epoch optimization screens plus one normalization screen: 1.5–3
  hours. Three locked seeds: 4.5–9 hours. With setup and fallback, budget
  8–16 GPU-hours total.
- Mount Drive and persist the repository, data, registry, `latest.pt`, and
  `best.pt` every epoch. Resume with the identical frozen config.

## Recommendation

Use Colab to preserve the paper's physical batch size 32. Run one sensitivity
job or locked seed per session. The GTX 1660 Ti remains a development or
explicitly labelled accumulated-batch fallback. Never evaluate the learned
model on test before the selected configuration is committed.
