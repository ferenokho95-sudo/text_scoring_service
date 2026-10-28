from collections import deque
from threading import Lock
import numpy as np
from app.model import FEATURES


def population_stability_index(reference, current, edges):
    """PSI with epsilon smoothing; edges always include both infinities."""
    p = np.histogram(reference, bins=edges)[0].astype(float)
    q = np.histogram(current, bins=edges)[0].astype(float)
    p = np.maximum(p / p.sum(), 1e-6)
    q = np.maximum(q / q.sum(), 1e-6)
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum((q - p) * np.log(q / p)))


class DriftMonitor:
    def __init__(self, reference, window_size=1000, min_samples=50, threshold=0.2):
        if not 1 <= min_samples <= window_size:
            raise ValueError("Expected 1 <= min_samples <= window_size")
        self.reference = np.asarray(reference, dtype=float)
        self.window = deque(maxlen=window_size)
        self.min_samples = min_samples
        self.threshold = threshold
        self.total_seen = 0
        self.lock = Lock()
        self.edges = []
        for col in self.reference.T:
            inner = np.unique(np.quantile(col, [0.1, 0.25, 0.5, 0.75, 0.9]))
            self.edges.append(np.r_[-np.inf, inner, np.inf])

    def add(self, rows):
        with self.lock:
            for row in rows:
                self.window.append(tuple(float(v) for v in row))
            self.total_seen += len(rows)

    def report(self):
        with self.lock:
            current = np.asarray(list(self.window), dtype=float)
            total = self.total_seen
        n = len(current)
        ready = n >= self.min_samples
        values = {}
        if ready:
            for i, name in enumerate(FEATURES):
                psi = population_stability_index(self.reference[:, i], current[:, i], self.edges[i])
                values[name] = {"psi": round(psi, 6), "drift_detected": psi >= self.threshold}
        return {"status": "ready" if ready else "insufficient_data", "samples": n,
                "total_seen": total, "window_size": self.window.maxlen,
                "min_samples": self.min_samples, "psi_threshold": self.threshold,
                "drift_detected": any(v["drift_detected"] for v in values.values()) if ready else None,
                "features": values}
