from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from app.drift import DriftMonitor, population_stability_index


def test_identical_distributions_have_zero_psi():
    x = np.arange(100)
    edges = [-np.inf, 25, 50, 75, np.inf]
    assert population_stability_index(x, np.tile(x, 3), edges) == pytest.approx(0)


def test_shift_detected_and_bounded_window():
    reference = np.tile(np.arange(100).reshape(-1, 1), (1, 5))
    monitor = DriftMonitor(reference, window_size=20, min_samples=10)
    monitor.add(np.full((30, 5), 1000))
    result = monitor.report()
    assert result["samples"] == 20
    assert result["total_seen"] == 30
    assert result["drift_detected"] is True


def test_small_window_does_not_claim_no_drift():
    monitor = DriftMonitor(np.zeros((60, 5)), min_samples=50)
    monitor.add(np.zeros((3, 5)))
    assert monitor.report()["drift_detected"] is None


def test_constant_baseline_has_finite_psi():
    monitor = DriftMonitor(np.zeros((60, 5)), min_samples=5)
    monitor.add(np.ones((5, 5)))
    assert all(np.isfinite(v["psi"]) for v in monitor.report()["features"].values())


def test_concurrent_updates():
    monitor = DriftMonitor(np.zeros((60, 5)), window_size=200, min_samples=5)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: monitor.add(np.ones((5, 5))), range(40)))
    assert monitor.report()["samples"] == 200
    assert monitor.report()["total_seen"] == 200


def test_invalid_window():
    with pytest.raises(ValueError):
        DriftMonitor(np.zeros((5, 5)), window_size=5, min_samples=10)
