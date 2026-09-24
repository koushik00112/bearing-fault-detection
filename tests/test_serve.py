"""Serving tested end to end with a tiny fixture model (the CI 'fixture model' check)."""

import json

import pytest
from fastapi.testclient import TestClient

from bearing import drift, models
from bearing.serve import app as serve


@pytest.fixture(scope="module")
def model_dir(ws, tmp_path_factory):
    d = tmp_path_factory.mktemp("model")
    m = models.make("rf", ("healthy", "inner", "outer"), ws.fs, n_estimators=10)
    m.fit(ws, None, seed=0)
    m.save(d)
    (d / "serving.json").write_text(
        json.dumps(
            {
                "window": ws.X.shape[2],
                "trained_on": {"dataset_source": "synthetic"},
                "drift_reference": drift.reference(ws.X, ws.channels),
            }
        )
    )
    return d


@pytest.fixture
def client(model_dir, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    with TestClient(serve.app) as c:
        yield c


def body(ws, idx, fs=None):
    return {
        "fs": fs or ws.fs,
        "windows": [
            {"channels": {ch: ws.X[i, c].tolist() for c, ch in enumerate(ws.channels)}} for i in idx
        ],
    }


def test_ready_and_model_info(client, ws):
    assert client.get("/readyz").json()["model_version"].startswith("rf-")
    info = client.get("/model").json()
    assert info["channels"] == list(ws.channels) and info["window"] == 2048


def test_predict_returns_probabilities_and_fault_probability(client, ws):
    r = client.post("/predict", json=body(ws, [0, 1, 2]))
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 3
    for p in preds:
        assert abs(sum(p["probabilities"].values()) - 1) < 1e-4
        assert p["fault_probability"] == pytest.approx(1 - p["probabilities"]["healthy"], abs=1e-5)
        assert p["predicted_class"] in ("healthy", "inner", "outer")


def test_serving_matches_offline_predictions(client, ws, model_dir):
    offline = models.load(model_dir).predict_proba(ws.X[:4])
    served = client.post("/predict", json=body(ws, range(4))).json()["predictions"]
    for row, p in zip(offline, served, strict=True):
        assert [p["probabilities"][c] for c in ("healthy", "inner", "outer")] == pytest.approx(
            row, abs=1e-5
        )


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda b: b.update(fs=12_000.0), "fs must be"),
        (lambda b: b["windows"][0]["channels"].pop("phase_current_2"), "channels must be"),
        (lambda b: b["windows"][0]["channels"].update(vibration_1=[0.0] * 100), "expected 2048"),
        (lambda b: b["windows"][0]["channels"]["vibration_1"].__setitem__(5, float("nan")), "NaN"),
    ],
)
def test_bad_requests_are_rejected_with_a_reason(client, ws, mutate, reason):
    b = body(ws, [0])
    mutate(b)
    # Raw JSON, since the client refuses to encode NaN but a real sender might.
    r = client.post("/predict", content=json.dumps(b), headers={"content-type": "application/json"})
    assert r.status_code == 422 and reason in r.text


def test_too_many_windows(client, ws):
    assert client.post("/predict", json=body(ws, range(65))).status_code == 422


def test_metrics_and_drift_after_traffic(client, ws):
    for start in range(0, 128, 32):
        client.post("/predict", json=body(ws, range(start, start + 32)))
    text = client.get("/metrics").text
    assert "predictions_total" in text and "prediction_duration_seconds" in text
    d = client.get("/drift").json()
    assert d["psi"] and d["threshold"] == 0.25


def test_no_model_means_not_ready(monkeypatch):
    monkeypatch.delenv("MODEL_DIR", raising=False)
    serve.state.model = None
    with TestClient(serve.app) as c:
        assert c.get("/readyz").status_code == 503
        assert c.get("/livez").status_code == 200
