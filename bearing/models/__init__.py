"""Models share one interface: fit on windows, predict class probabilities from raw windows.

Keeping feature extraction and normalisation *inside* the model means the serving path
calls exactly the same code as evaluation, and nothing is fitted on test data.
"""

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from bearing.windowing import WindowSet


class Classifier(Protocol):
    name: str
    classes: tuple[str, ...]

    def fit(self, train: WindowSet, val: WindowSet | None, seed: int) -> None: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...

    def save(self, directory: Path) -> None: ...


def make(name: str, classes: tuple[str, ...], fs: float, **kwargs: Any) -> Classifier:
    if name in ("rf", "hgb"):
        from bearing.models.baseline import FeatureModel

        return FeatureModel(name, classes, fs, **kwargs)
    if name == "cnn":
        from bearing.models.cnn import CNNModel

        return CNNModel(classes, fs, **kwargs)
    raise ValueError(f"unknown model {name}")


def load(directory: Path) -> Classifier:
    meta = json.loads((directory / "meta.json").read_text())
    if meta["model"] in ("rf", "hgb"):
        from bearing.models.baseline import FeatureModel

        return FeatureModel.load(directory)
    if meta["model"] == "cnn":
        from bearing.models.cnn import CNNModel

        return CNNModel.load(directory)
    raise ValueError(f"unknown model type in {directory}: {meta['model']}")
