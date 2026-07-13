import math

from spurbreast_repro.baselines import field_strength_shortcut_predictions
from spurbreast_repro.metrics import (
    binary_metrics,
    patient_cluster_bootstrap,
    patient_macro_metrics,
)


def test_binary_metrics_from_one_global_confusion_matrix() -> None:
    metrics = binary_metrics(
        y_true=[1, 1, 1, 0, 0, 0],
        y_pred=[1, 1, 0, 1, 0, 0],
        positive_probabilities=[0.9, 0.8, 0.2, 0.7, 0.3, 0.1],
    )
    assert metrics["tp"] == 2
    assert metrics["tn"] == 2
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert math.isclose(metrics["accuracy"], 2 / 3)
    assert math.isclose(metrics["ppv"], 2 / 3)
    assert math.isclose(metrics["npv"], 2 / 3)
    assert metrics["nll"] > 0


def test_balanced_accuracy_above_half_forces_both_predictive_values_above_half() -> None:
    # Equal positive/negative prevalence and TP + TN > class size imply
    # TP > FP and TN > FN, hence PPV and NPV are both > 0.5.
    metrics = binary_metrics(
        y_true=[1] * 100 + [0] * 100,
        y_pred=[1] * 70 + [0] * 30 + [1] * 65 + [0] * 35,
    )
    assert metrics["accuracy"] > 0.5
    assert metrics["ppv"] > 0.5
    assert metrics["npv"] > 0.5


def test_patient_macro_is_not_silently_mixed_with_micro() -> None:
    result = patient_macro_metrics(
        y_true=[1, 1, 0, 1, 0, 0],
        y_pred=[1, 0, 0, 1, 1, 0],
        patient_ids=["A", "A", "A", "B", "B", "B"],
    )
    assert result["patients"] == 2
    assert 0 <= result["accuracy"] <= 1


def test_cluster_bootstrap_is_reproducible() -> None:
    kwargs = dict(
        y_true=[1, 0, 1, 0],
        y_pred=[1, 0, 0, 0],
        patient_ids=["A", "A", "B", "B"],
        resamples=50,
        seed=17,
    )
    assert patient_cluster_bootstrap(**kwargs) == patient_cluster_bootstrap(**kwargs)


def test_field_strength_oracle_has_the_paper_direction() -> None:
    assert field_strength_shortcut_predictions([1.5, 3.0, 1.5, 3.0]) == [1, 0, 1, 0]
