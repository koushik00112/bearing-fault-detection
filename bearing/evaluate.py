"""Metrics, noise robustness, latency and size."""

import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from bearing.models import Classifier


def classification_metrics(
    y_true: np.ndarray, proba: np.ndarray, classes: tuple[str, ...]
) -> dict[str, Any]:
    y_pred = proba.argmax(axis=1)
    labels = list(range(len(classes)))
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    present = sorted(set(y_true.tolist()))
    return {
        # Macro-F1 over the classes present in this test set: a class that can't appear
        # (no test examples) shouldn't count as a zero.
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)
        ),
        "accuracy": float(np.mean(y_true == y_pred)),
        "per_class_recall": {
            c: float(r) for c, r, i in zip(classes, recalls, labels, strict=True) if i in present
        },
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n_test": int(len(y_true)),
    }


def add_noise(X: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    """White Gaussian noise at a given SNR, per window and channel (signal power measured)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float32)
    power = np.mean(X**2, axis=2, keepdims=True)
    noise_power = power / (10 ** (snr_db / 10))
    return (X + rng.standard_normal(X.shape).astype(np.float32) * np.sqrt(noise_power)).astype(
        np.float32
    )


def noise_curve(
    model: Classifier,
    X: np.ndarray,
    y: np.ndarray,
    snrs_db: tuple[float, ...],
    seed: int,
) -> dict[str, float]:
    out = {"clean": classification_metrics(y, model.predict_proba(X), model.classes)["macro_f1"]}
    for snr in snrs_db:
        noisy = add_noise(X, snr, seed)
        out[f"{snr:g}dB"] = classification_metrics(y, model.predict_proba(noisy), model.classes)[
            "macro_f1"
        ]
    return out


def latency_ms(model: Classifier, X: np.ndarray, n: int = 200) -> dict[str, float]:
    """Single-window inference time (the serving case), including feature extraction."""
    n = min(n, len(X))
    model.predict_proba(X[:1])  # warm-up
    times = []
    for i in range(n):
        start = time.perf_counter()
        model.predict_proba(X[i : i + 1])
        times.append((time.perf_counter() - start) * 1000)
    arr = np.array(times)
    return {"p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95)), "n": n}


def model_size_bytes(model: Classifier) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        model.save(Path(tmp))
        return sum(p.stat().st_size for p in Path(tmp).iterdir() if p.name != "meta.json")


def mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n": int(len(arr)),
    }
