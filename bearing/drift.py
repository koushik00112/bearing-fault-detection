"""Input drift monitoring with the population stability index (PSI).

At export time we store, for a few cheap per-window features, the training distribution
as quantile bin edges. At serving time a rolling window of recent requests is binned the
same way and compared. Rule of thumb: PSI < 0.1 stable, 0.1-0.25 moderate shift, > 0.25
major shift. These thresholds are conventions, not guarantees, so treat an alert as
"look at the data", not "the model is wrong".
"""

from collections import deque

import numpy as np

MONITORED = ("rms", "kurtosis", "crest")
N_BINS = 10


def window_stats(X: np.ndarray) -> dict[str, np.ndarray]:
    """Cheap per-window, per-channel stats: X (n, C, L) -> {name: (n, C)}."""
    X = np.asarray(X, dtype=np.float64)
    rms = np.sqrt(np.mean(X**2, axis=2))
    centred = X - X.mean(axis=2, keepdims=True)
    var = np.mean(centred**2, axis=2) + 1e-12
    kurt = np.mean(centred**4, axis=2) / var**2 - 3
    crest = np.max(np.abs(X), axis=2) / (rms + 1e-12)
    return {"rms": rms, "kurtosis": kurt, "crest": crest}


def reference(X: np.ndarray, channels: tuple[str, ...]) -> dict[str, dict[str, list[float]]]:
    """Training-set bin edges and expected bin proportions for each monitored feature."""
    stats = window_stats(X)
    ref: dict[str, dict[str, list[float]]] = {}
    for name in MONITORED:
        for c, ch in enumerate(channels):
            values = stats[name][:, c]
            edges = np.unique(np.quantile(values, np.linspace(0, 1, N_BINS + 1)))
            counts = np.histogram(np.clip(values, edges[0], edges[-1]), bins=edges)[0]
            ref[f"{ch}.{name}"] = {
                "edges": edges.tolist(),
                "expected": (counts / counts.sum()).tolist(),
            }
    return ref


def psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-4) -> float:
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


class RollingDrift:
    """Keeps the last `size` windows' stats and computes PSI against the reference."""

    def __init__(
        self,
        ref: dict[str, dict[str, list[float]]],
        channels: tuple[str, ...],
        size: int = 500,
        min_samples: int = 100,
    ) -> None:
        self.ref = ref
        self.channels = channels
        self.min_samples = min_samples
        self.buffers: dict[str, deque[float]] = {k: deque(maxlen=size) for k in ref}

    def update(self, X: np.ndarray) -> None:
        stats = window_stats(X)
        for name in MONITORED:
            for c, ch in enumerate(self.channels):
                key = f"{ch}.{name}"
                if key in self.buffers:
                    self.buffers[key].extend(stats[name][:, c].tolist())

    def scores(self) -> dict[str, float]:
        """PSI per feature, or {} until enough samples have arrived."""
        out = {}
        for key, buf in self.buffers.items():
            if len(buf) < self.min_samples:
                continue
            edges = np.asarray(self.ref[key]["edges"])
            # Out-of-range values land in the outer bins, so a shifted mean still registers.
            counts = np.histogram(np.clip(np.asarray(buf), edges[0], edges[-1]), bins=edges)[0]
            out[key] = psi(np.asarray(self.ref[key]["expected"]), counts / counts.sum())
        return out
