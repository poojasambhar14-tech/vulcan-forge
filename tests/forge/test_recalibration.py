import numpy as np

from vulcan.forge.recalibration import fit_temperature, recalibrate, make_recalibrated_predict_fn
from vulcan.drift.detectors import expected_calibration_error


def test_fit_temperature_improves_ece_on_synthetically_miscalibrated_probs():
    """Construct probs that are a known monotonic (overconfident) distortion
    of the true probabilities, and confirm temperature scaling recovers
    most of the calibration error."""
    rng = np.random.default_rng(0)
    n = 2000
    true_p = rng.uniform(0.1, 0.9, size=n)
    labels = (rng.random(n) < true_p).astype(float)
    # overconfident distortion: push probabilities away from 0.5
    logits = np.log(true_p / (1 - true_p))
    overconfident_probs = 1 / (1 + np.exp(-logits * 2.5))  # true temperature = 2.5

    ece_before = expected_calibration_error(overconfident_probs, labels)
    t = fit_temperature(overconfident_probs, labels)
    calibrated = 1 / (1 + np.exp(-(np.log(overconfident_probs / (1 - overconfident_probs))) / t))
    ece_after = expected_calibration_error(calibrated, labels)

    assert ece_after < ece_before
    assert 2.0 < t < 3.0  # should recover something close to the true 2.5 distortion


def test_recalibrate_end_to_end_with_predict_fn():
    rng = np.random.default_rng(1)
    n = 500
    true_p = rng.uniform(0.2, 0.8, size=n)
    labels = (rng.random(n) < true_p).astype(float)
    logits = np.log(true_p / (1 - true_p))
    overconfident = 1 / (1 + np.exp(-logits * 3.0))

    records = [{"outcome_success": bool(labels[i]), "_idx": i} for i in range(n)]
    predict_fn = lambda r: float(overconfident[r["_idx"]])

    result = recalibrate(predict_fn, records)
    assert result.ece_after <= result.ece_before
    # Temperature is now fit on an earlier split and REPORTED on a disjoint,
    # strictly later split, so the reported improvement is held-out rather
    # than measured on the data the parameter was tuned to.
    assert result.n_fit_examples + result.n_calibration_examples == n
    assert result.n_fit_examples > 0 and result.n_calibration_examples > 0
    assert abs(result.accuracy_after - result.accuracy_before) < 0.05  # recalibration shouldn't change argmax much


def test_recalibrated_predict_fn_wraps_correctly():
    base_fn = lambda r: 0.9
    wrapped = make_recalibrated_predict_fn(base_fn, temperature=2.0)
    p = wrapped({})
    # temperature > 1 should pull an extreme probability toward 0.5
    assert 0.5 < p < 0.9
