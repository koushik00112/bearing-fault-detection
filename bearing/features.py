"""Hand-engineered features per window and channel: the baseline models' input."""

import numpy as np
from scipy.stats import kurtosis, skew

N_BANDS = 8
FEATURES_PER_CHANNEL = 10 + (2 + N_BANDS) + 2  # time + frequency + envelope


def _time_features(x: np.ndarray) -> np.ndarray:
    """x: (n, L) -> (n, 10)."""
    eps = 1e-12
    mean_abs = np.mean(np.abs(x), axis=1)
    rms = np.sqrt(np.mean(x**2, axis=1))
    peak = np.max(np.abs(x), axis=1)
    std = np.std(x, axis=1)
    sqrt_mean = np.mean(np.sqrt(np.abs(x)), axis=1) ** 2
    return np.column_stack(
        [
            rms,
            std,
            peak,
            np.ptp(x, axis=1),
            kurtosis(x, axis=1, fisher=True),
            skew(x, axis=1),
            peak / (rms + eps),  # crest factor
            rms / (mean_abs + eps),  # shape factor
            peak / (mean_abs + eps),  # impulse factor
            peak / (sqrt_mean + eps),  # clearance factor
        ]
    )


def _freq_features(x: np.ndarray, fs: float) -> np.ndarray:
    """Spectral centroid, spread, and log energy in N_BANDS log-spaced bands: (n, 2 + N_BANDS)."""
    n = x.shape[1]
    spec = np.abs(np.fft.rfft(x * np.hanning(n), axis=1)) ** 2
    freqs = np.fft.rfftfreq(n, 1 / fs)
    total = spec.sum(axis=1) + 1e-12
    centroid = (spec * freqs).sum(axis=1) / total
    spread = np.sqrt((spec * (freqs - centroid[:, None]) ** 2).sum(axis=1) / total)
    edges = np.geomspace(10, fs / 2, N_BANDS + 1)
    bands = [
        np.log10(spec[:, (freqs >= lo) & (freqs < hi)].sum(axis=1) + 1e-12)
        for lo, hi in zip(edges[:-1], edges[1:], strict=True)
    ]
    return np.column_stack([centroid, spread, *bands])


def _envelope_features(x: np.ndarray, fs: float) -> np.ndarray:
    """Kurtosis and peak-to-mean ratio of the envelope spectrum below 500 Hz: (n, 2).

    Bearing faults show up as periodic impacts, i.e. peaks in the envelope spectrum at the
    fault frequencies. This summarises how 'peaky' it is without needing shaft speed.
    """
    from scipy.signal import hilbert

    env = np.abs(hilbert(x - x.mean(axis=1, keepdims=True), axis=1))
    env -= env.mean(axis=1, keepdims=True)
    spec = np.abs(np.fft.rfft(env, axis=1))
    freqs = np.fft.rfftfreq(x.shape[1], 1 / fs)
    low = spec[:, (freqs > 2) & (freqs < 500)]
    return np.column_stack([kurtosis(low, axis=1), low.max(axis=1) / (low.mean(axis=1) + 1e-12)])


def feature_names(channels: tuple[str, ...]) -> list[str]:
    time = [
        "rms",
        "std",
        "peak",
        "ptp",
        "kurtosis",
        "skew",
        "crest",
        "shape",
        "impulse",
        "clearance",
    ]
    freq = ["centroid", "spread", *[f"band{i}" for i in range(N_BANDS)]]
    env = ["env_kurtosis", "env_peak_ratio"]
    return [f"{c}.{f}" for c in channels for f in (*time, *freq, *env)]


def extract(X: np.ndarray, fs: float, batch: int = 2048) -> np.ndarray:
    """(n, channels, L) -> (n, channels * FEATURES_PER_CHANNEL) float32.

    Works in batches, so X can be a memory-mapped array larger than RAM.
    """
    out = []
    for i in range(0, len(X), batch):
        xb = np.asarray(X[i : i + batch], dtype=np.float64)
        per_channel = [
            np.column_stack(
                [
                    _time_features(xb[:, c]),
                    _freq_features(xb[:, c], fs),
                    _envelope_features(xb[:, c], fs),
                ]
            )
            for c in range(xb.shape[1])
        ]
        out.append(np.column_stack(per_channel))
    if not out:
        return np.empty((0, X.shape[1] * FEATURES_PER_CHANNEL), dtype=np.float32)
    F = np.concatenate(out).astype(np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
