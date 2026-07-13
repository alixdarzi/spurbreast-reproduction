# Gate 3 smoke-test evidence

Status: passed on 2026-07-13. No test-set DataLoader was constructed.

| Check | Evidence |
|---|---|
| Regression suite | 14 passed in 2.30 s |
| Run ID | `smoke_test-seed2025-20260713T091547Z` |
| Git commit recorded by run | `e3ecc5cd3c2f55d2fb97e7d66fef6ae863fc4f5d` |
| Dataset MD5 | `dbe61da7dc7b06c69c10dbbea0a13b40` |
| Environment | Python 3.11.15; Torch 2.0.1+cpu; Windows 10 |
| Pause | epoch 0; 8.046 s measured runtime |
| Resume | loaded `latest.pt`, completed epoch 1; 7.843 s |
| Sample | 8 train and 8 label-balanced validation images per epoch |
| Checkpoints | best 282,618,173 B; latest 282,619,787 B |

The metrics are not scientific estimates: the randomly initialized model saw
only two batches and no test images. The run verifies real PNG decoding,
transforms, forward/backward passes, strict metric JSON, atomic checkpoints,
saved optimizer/scaler/RNG/DataLoader state, config matching, and resume.

An earlier smoke exposed a single-label capped validation prefix. The cap now
uses a deterministic label-interleaved subset, with a regression test. The old
run remains in `experiments/registry.csv` for provenance.
