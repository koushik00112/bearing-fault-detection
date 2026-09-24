"""Engineered features + random forest or gradient boosting."""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bearing import features
from bearing.windowing import WindowSet


class FeatureModel:
    def __init__(self, name: str, classes: tuple[str, ...], fs: float, **params: Any) -> None:
        self.name = name
        self.classes = tuple(classes)
        self.fs = fs
        self.params = params
        self.pipeline: Pipeline | None = None
        self.channels: tuple[str, ...] = ()

    def _estimator(self, seed: int) -> Any:
        if self.name == "rf":
            return RandomForestClassifier(
                n_estimators=self.params.get("n_estimators", 300),
                min_samples_leaf=self.params.get("min_samples_leaf", 2),
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed,
            )
        return HistGradientBoostingClassifier(
            max_iter=self.params.get("max_iter", 300),
            learning_rate=self.params.get("learning_rate", 0.1),
            class_weight="balanced",
            early_stopping=False,
            random_state=seed,
        )

    def fit(self, train: WindowSet, val: WindowSet | None, seed: int) -> None:
        # Validation data isn't needed: no early stopping, fixed hyperparameters.
        self.channels = train.channels
        y = np.array([self.classes.index(label) for label in train.meta["label"]])
        self.pipeline = Pipeline([("scale", StandardScaler()), ("clf", self._estimator(seed))])
        self.pipeline.fit(features.extract(train.X, self.fs), y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("model is not fitted")
        proba = self.pipeline.predict_proba(features.extract(X, self.fs))
        # A class absent from training gets no column; put it back as zeros.
        full = np.zeros((len(X), len(self.classes)), dtype=np.float64)
        full[:, self.pipeline.classes_] = proba
        return full

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, directory / "model.joblib")
        (directory / "meta.json").write_text(
            json.dumps(
                {
                    "model": self.name,
                    "classes": list(self.classes),
                    "fs": self.fs,
                    "channels": list(self.channels),
                    "params": self.params,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, directory: Path) -> "FeatureModel":
        meta = json.loads((directory / "meta.json").read_text())
        model = cls(meta["model"], tuple(meta["classes"]), meta["fs"], **meta["params"])
        model.channels = tuple(meta["channels"])
        model.pipeline = joblib.load(directory / "model.joblib")  # noqa: S301 - own artefact
        return model
