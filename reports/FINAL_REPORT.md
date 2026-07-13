# Final report

Results will be written only after the data audit, smoke tests, Approval Gate 3,
and locked reproduction runs are complete.

The released-data audit passed: 19,926 valid 320×320 grayscale PNGs, 700
unique patients, no corruption, no duplicate hashes, and no patient leakage.
The deterministic field-strength oracle is perfect on training and validation
but gives 0.457 accuracy, 0.453 PPV, and 0.461 NPV on the released test set.
That audit result is not a model reproduction and exposes an unresolved
disagreement with Table 2; see `docs/DATA_AUDIT.md`.

Gate 3 engineering checks passed: 14 tests and a real-data, two-epoch CPU smoke
run that paused after epoch 0 and resumed from `latest.pt`. Expensive screening,
locked training, and learned-model test evaluation remain pending approval.

## Paper versus reproduction

| Split | Metric | Paper | Reproduction | Difference | 95% patient-cluster CI |
|---|---|---:|---:|---:|---|
| Training | Accuracy | 0.99 | Pending | Pending | Pending |
| Training | PPV | 0.98 | Pending | Pending | Pending |
| Training | NPV | 0.99 | Pending | Pending | Pending |
| Validation | Accuracy | 0.99 | Pending | Pending | Pending |
| Validation | PPV | 1.00 | Pending | Pending | Pending |
| Validation | NPV | 0.98 | Pending | Pending | Pending |
| Test | Accuracy | 0.52 | Pending | Pending | Pending |
| Test | PPV | 0.62 | Pending | Pending | Pending |
| Test | NPV | 0.41 | Pending | Pending | Pending |
