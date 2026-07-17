# Objective success criteria

## Mandatory validity

- Archive and manifest hashes match the Gate 3 audit.
- Exact released patient-disjoint folders are used.
- No test-based model, threshold, preprocessing, calibration, or seed choice.
- Every prespecified seed and failed run is reported.
- Primary metrics use one pooled slice confusion matrix; patient-macro and
  patient-cluster results remain separate.

## Close numerical reproduction

For the mean of seeds 2025–2027:

- training and validation accuracy within 0.03 of Table 2, with PPV and NPV
  within 0.05 where defined;
- test accuracy within 0.05 of 0.52; and
- validation-to-test accuracy drop at least 0.30, with test predicted-positive
  rate at least 0.50 higher at 1.5 T than at 3 T.

The paper's balanced-test 0.52 accuracy, 0.62 PPV, and 0.41 NPV cannot coexist
under the primary global definitions. They therefore cannot all be mandatory.
Meeting test accuracy and shortcut-direction criteria with a larger PPV/NPV
difference counts as a credible reproduction of the phenomenon with an
unresolved metric-provenance discrepancy, not an exact Table 2 reproduction.

A valid run outside these bands is informative but is labelled an attempted,
not successful close numerical, reproduction.

## Final outcome

The locked three-seed mean met every mandatory validity rule and every
prespecified phenomenon-level criterion:

- training accuracy 0.9998 and validation accuracy 0.9893;
- test accuracy 0.5175 (difference from Table 2: −0.0025);
- validation-to-test accuracy drop 0.4717; and
- test predicted-positive-rate gap 0.8535 between 1.5 T and 3 T.

Training/validation PPV and NPV are within their numerical bands. Test PPV and
NPV differ by −0.0985 and +0.1048, respectively, and are classified under the
predeclared metric-provenance exception. The final label is therefore
**successful close reproduction of the field-strength shortcut phenomenon**,
not exact reproduction of all three test metrics.
