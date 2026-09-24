"""Model-serving API.

  MODEL_DIR=results/paderborn/models/B2_bearing/hgb uvicorn bearing.serve.app:app --port 8100

POST /predict takes raw windows (the same shape the model was trained on), returns class
probabilities, and updates drift statistics. /metrics exposes latency, prediction counts
and PSI drift scores for Prometheus.
"""

import hashlib
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from bearing import models
from bearing.drift import RollingDrift

log = logging.getLogger("serve")

PRED_LATENCY = Histogram(
    "prediction_duration_seconds",
    "Time to score one request",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1),
)
PREDICTIONS = Counter("predictions_total", "Windows scored", ["predicted_class"])
REJECTED = Counter("prediction_requests_rejected_total", "Invalid requests", ["reason"])
DRIFT_PSI = Gauge("input_drift_psi", "PSI of recent inputs vs training data", ["feature"])
DRIFT_THRESHOLD = 0.25
MAX_WINDOWS = 64


class Window(BaseModel):
    channels: dict[str, list[float]]


class PredictRequest(BaseModel):
    fs: float = Field(gt=0)
    windows: list[Window] = Field(min_length=1, max_length=MAX_WINDOWS)


class Prediction(BaseModel):
    predicted_class: str
    probabilities: dict[str, float]
    fault_probability: float = Field(description="1 - P(healthy)")


class PredictResponse(BaseModel):
    model_version: str
    predictions: list[Prediction]


class State:
    model: Any = None
    serving: dict[str, Any] = {}
    version: str = ""
    drift: RollingDrift | None = None


state = State()


def load_model(directory: Path) -> None:
    state.model = models.load(directory)
    state.serving = json.loads((directory / "serving.json").read_text())
    digest = hashlib.sha256()
    for p in sorted(directory.iterdir()):
        if p.suffix in (".joblib", ".pt", ".npz"):
            digest.update(p.read_bytes())
    state.version = f"{state.model.name}-{digest.hexdigest()[:12]}"
    state.drift = RollingDrift(state.serving["drift_reference"], state.model.channels)
    log.info("loaded model %s from %s", state.version, directory)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    model_dir = os.environ.get("MODEL_DIR")
    if model_dir:
        load_model(Path(model_dir))
    yield


app = FastAPI(title="Bearing fault model", version="0.1.0", lifespan=lifespan)


def to_array(req: PredictRequest) -> np.ndarray:
    channels = state.model.channels
    length = state.serving["window"]
    if abs(req.fs - state.model.fs) > 1e-6:
        REJECTED.labels("sampling_rate").inc()
        raise HTTPException(422, f"fs must be {state.model.fs} Hz (got {req.fs})")
    X = np.empty((len(req.windows), len(channels), length), dtype=np.float32)
    for i, w in enumerate(req.windows):
        if set(w.channels) != set(channels):
            REJECTED.labels("channels").inc()
            raise HTTPException(422, f"window {i}: channels must be {list(channels)}")
        for c, name in enumerate(channels):
            values = w.channels[name]
            if len(values) != length:
                REJECTED.labels("length").inc()
                raise HTTPException(422, f"window {i}/{name}: expected {length} samples")
            X[i, c] = values
    if not np.isfinite(X).all():
        REJECTED.labels("non_finite").inc()
        raise HTTPException(422, "windows contain NaN or infinite values")
    return X


@app.post("/predict")
def predict(req: PredictRequest) -> PredictResponse:
    if state.model is None:
        raise HTTPException(503, "no model loaded")
    X = to_array(req)
    start = time.perf_counter()
    proba = state.model.predict_proba(X)
    PRED_LATENCY.observe(time.perf_counter() - start)

    classes = list(state.model.classes)
    healthy = classes.index("healthy") if "healthy" in classes else None
    out = []
    for p in proba:
        cls = classes[int(np.argmax(p))]
        PREDICTIONS.labels(cls).inc()
        out.append(
            Prediction(
                predicted_class=cls,
                probabilities={c: round(float(v), 6) for c, v in zip(classes, p, strict=True)},
                fault_probability=round(1.0 - float(p[healthy]), 6) if healthy is not None else 0.0,
            )
        )

    assert state.drift is not None
    state.drift.update(X)
    scores = state.drift.scores()
    for feature, value in scores.items():
        DRIFT_PSI.labels(feature).set(value)
    if scores and max(scores.values()) > DRIFT_THRESHOLD:
        worst = max(scores, key=lambda k: scores[k])
        log.warning("input drift: %s PSI=%.2f", worst, scores[worst])
    return PredictResponse(model_version=state.version, predictions=out)


@app.get("/livez")
def livez() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict[str, str]:
    if state.model is None:
        raise HTTPException(503, "no model loaded")
    return {"status": "ready", "model_version": state.version}


@app.get("/model")
def model_info() -> dict[str, Any]:
    if state.model is None:
        raise HTTPException(503, "no model loaded")
    return {
        "model_version": state.version,
        "classes": list(state.model.classes),
        "channels": list(state.model.channels),
        "fs": state.model.fs,
        "window": state.serving["window"],
        "trained_on": state.serving.get("trained_on"),
    }


@app.get("/drift")
def drift_scores() -> dict[str, Any]:
    scores = state.drift.scores() if state.drift else {}
    return {
        "threshold": DRIFT_THRESHOLD,
        "psi": scores,
        "drifting": sorted(k for k, v in scores.items() if v > DRIFT_THRESHOLD),
    }


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
