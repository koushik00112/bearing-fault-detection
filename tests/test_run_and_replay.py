import json
from argparse import Namespace

import httpx
import numpy as np

from bearing import prepare, replay
from bearing.data import synthetic
from bearing.data.paderborn import CHANNELS
from bearing.report import write_report
from bearing.run import Config, run
from bearing.tracking import Tracker


def test_full_protocol_on_synthetic_data(tmp_path):
    recs = synthetic.make_dataset(bearings_per_class=3, reps=3, seconds=0.5, seed=2)
    data = tmp_path / "data"
    ws = prepare.write_dataset(
        iter(recs), data, CHANNELS, 2048, 1024, None, {"source": "synthetic"}
    )
    cfg = Config(
        models=("rf",),
        seeds=(0, 1),
        snrs_db=(10, 0),
        healthy_test_bearings=synthetic.HEALTHY_TEST_BEARINGS,
        model_params={"rf": {"n_estimators": 20}},
    )
    out = tmp_path / "out"
    result = run(ws, cfg, out, Tracker(out, "t", use_mlflow=False))
    result["run_info"] = {"dataset": {"source": "synthetic"}, "config": {"seeds": [0, 1]}}

    scenarios = {r["scenario"] for r in result["rows"]}
    assert scenarios == set(cfg.scenarios)
    assert all(r["leaky"] == (r["scenario"] == "A_leaky_window") for r in result["rows"])
    assert sum(r["scenario"] == "C_loco" for r in result["rows"]) == 4 * 2  # folds x seeds

    # The exported model's held-out recordings really were held out.
    export = out / "models" / "B2_bearing" / "rf"
    held_out = set(json.loads((export / "held_out_recordings.json").read_text()))
    serving = json.loads((export / "serving.json").read_text())
    assert serving["trained_on"]["scenario"] == "B2_bearing"
    assert held_out and held_out.isdisjoint(set())  # non-empty
    report = write_report(result, out).read_text()
    assert "SYNTHETIC DATA" in report and "leaky, deliberately" in report
    assert (out / "confusion.png").exists() and (out / "noise_curve.png").exists()


def test_prepared_dataset_round_trips_through_disk(tmp_path, recordings):
    ws = prepare.write_dataset(
        iter(recordings[:6]), tmp_path, CHANNELS, 2048, 1024, 4, {"source": "synthetic"}
    )
    again = prepare.load(tmp_path)
    np.testing.assert_array_equal(np.asarray(again.X), np.asarray(ws.X))
    assert list(again.meta["recording_id"]) == list(ws.meta["recording_id"])


def test_replay_posts_fault_probabilities_to_the_platform(tmp_path, recordings):
    ws = prepare.write_dataset(
        iter(recordings[:4]), tmp_path / "d", CHANNELS, 2048, 1024, 3, {"source": "synthetic"}
    )
    held = sorted(ws.meta["recording_id"].unique())[:2]
    (tmp_path / "held.json").write_text(json.dumps(held))
    calls = {"rules": 0, "devices": [], "batches": []}

    def platform(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/alert-rules" and req.method == "GET":
            return httpx.Response(200, json=[])
        if req.url.path == "/alert-rules":
            calls["rules"] += 1
            return httpx.Response(201, json={})
        if req.url.path == "/devices":
            calls["devices"].append(json.loads(req.content)["name"])
            return httpx.Response(201, json={"api_key": "tp_k"})
        calls["batches"].append(json.loads(req.content))
        return httpx.Response(200, json={"received": 3, "inserted": 3, "duplicates": 0})

    def model(req: httpx.Request) -> httpx.Response:
        n = len(json.loads(req.content)["windows"])
        return httpx.Response(200, json={"predictions": [{"fault_probability": 0.9}] * n})

    args = Namespace(
        data=tmp_path / "d",
        held_out=tmp_path / "held.json",
        admin_token="t",
        threshold=0.8,
        max_recordings=10,
        pause=0,
        device_prefix="replay",
        cache=tmp_path / "cache.json",
    )
    with (
        httpx.Client(base_url="http://m", transport=httpx.MockTransport(model)) as m,
        httpx.Client(base_url="http://p", transport=httpx.MockTransport(platform)) as p,
    ):
        summary = replay.replay(args, m, p)

    assert calls["rules"] == 1
    assert all(name.startswith("replay-synthetic-") for name in calls["devices"])
    readings = [r for b in calls["batches"] for r in b["readings"]]
    assert {r["metric"] for r in readings} == {"bearing_fault_prob"}
    per_device_ts = [r["ts"] for r in calls["batches"][0]["readings"]]
    assert per_device_ts == sorted(per_device_ts)
    assert all(v == 0.9 for v in summary.values())
