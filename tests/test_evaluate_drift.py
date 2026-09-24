import numpy as np
import pytest

from bearing import drift
from bearing.evaluate import add_noise, classification_metrics, mean_std


@pytest.mark.parametrize("snr", [20, 0, -5])
def test_add_noise_hits_the_requested_snr(snr):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 2, 1024)).astype(np.float32) * 3
    noisy = add_noise(X, snr, seed=1)
    measured = 10 * np.log10(np.mean(X**2) / np.mean((noisy - X) ** 2))
    assert abs(measured - snr) < 0.3


def test_macro_f1_ignores_classes_absent_from_test():
    y = np.array([0, 0, 2, 2])
    proba = np.eye(3)[[0, 0, 2, 2]]
    m = classification_metrics(y, proba, ("healthy", "inner", "outer"))
    assert m["macro_f1"] == 1.0
    assert "inner" not in m["per_class_recall"]
    assert m["confusion"][1] == [0, 0, 0]


def test_mean_std():
    assert mean_std([1.0, 3.0]) == {"mean": 2.0, "std": pytest.approx(np.sqrt(2)), "n": 2}
    assert mean_std([1.0])["std"] == 0.0


def test_psi_zero_for_same_distribution_and_large_for_a_shift():
    e = np.full(10, 0.1)
    assert drift.psi(e, e) == pytest.approx(0)
    shifted = np.array([0.0] * 5 + [0.2] * 5)
    assert drift.psi(e, shifted) > 0.25


def test_rolling_drift_flags_a_gain_change(ws):
    ref = drift.reference(ws.X, ws.channels)
    # A random sample with the same class mix as the reference.
    sample = np.sort(np.random.default_rng(0).choice(len(ws), 300, replace=False))
    same = drift.RollingDrift(ref, ws.channels, size=300, min_samples=100)
    assert same.scores() == {}
    same.update(ws.X[sample])
    assert max(same.scores().values()) < 0.25

    louder = drift.RollingDrift(ref, ws.channels, size=300, min_samples=100)
    louder.update(ws.X[sample] * 3)  # e.g. a sensor gain change
    assert louder.scores()["vibration_1.rms"] > 0.25


def test_a_change_in_fault_rate_also_shows_as_drift(ws):
    # Only healthy windows vs an all-class reference: PSI rises. Expected, and worth
    # knowing: drift alerts fire on real changes in the fleet, not just broken sensors.
    ref = drift.reference(ws.X, ws.channels)
    healthy_only = drift.RollingDrift(ref, ws.channels, size=200, min_samples=100)
    healthy_only.update(ws.X[np.flatnonzero(ws.meta["label"] == "healthy")[:200]])
    assert healthy_only.scores()["vibration_1.rms"] > 0.25
