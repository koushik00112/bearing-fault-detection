"""Stream held-out recordings through the model service into the telemetry platform.

  python -m bearing.replay --data data/processed/paderborn \
      --held-out results/paderborn/models/B2_bearing/hgb/held_out_recordings.json \
      --model-url http://localhost:8100 --platform-url http://localhost:8000

For each held-out recording (never seen in training), windows are sent to POST /predict,
and the returned fault probability is posted to the platform as the metric
`bearing_fault_prob` for a device named `replay-<source>-<bearing>`. A platform alert
rule (`bearing_fault_prob gt 0.8` by default) turns sustained high scores into alerts.

This is a replay of recorded data, not a live sensor, and every device name says so.
"""

import argparse
import json
import time
import uuid
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from itertools import zip_longest
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from bearing.prepare import load

METRIC = "bearing_fault_prob"


def ensure_rule(platform: httpx.Client, admin: dict[str, str], threshold: float) -> None:
    rules = platform.get("/alert-rules", headers=admin).raise_for_status().json()
    if not any(
        r["metric"] == METRIC and r["operator"] == "gt" and r["threshold"] == threshold
        for r in rules
    ):
        platform.post(
            "/alert-rules",
            headers=admin,
            json={"metric": METRIC, "operator": "gt", "threshold": threshold},
        ).raise_for_status()


def device_key(
    platform: httpx.Client, admin: dict[str, str], name: str, cache: dict[str, str]
) -> str:
    if name not in cache:
        r = platform.post("/devices", json={"name": name}, headers=admin)
        if r.status_code == 409:
            raise SystemExit(
                f"{name} already exists on the platform but its key isn't cached; "
                "delete the cache entry's device or use --device-prefix"
            )
        cache[name] = r.raise_for_status().json()["api_key"]
    return cache[name]


def replay(args: argparse.Namespace, model: httpx.Client, platform: httpx.Client) -> dict[str, Any]:
    ws = load(args.data)
    held_out = set(json.loads(args.held_out.read_text()))
    rows = ws.meta[ws.meta["recording_id"].isin(held_out)]
    if rows.empty:
        raise SystemExit("none of the held-out recordings are in this dataset")
    admin = {"X-Admin-Token": args.admin_token}
    ensure_rule(platform, admin, args.threshold)
    cache: dict[str, str] = json.loads(args.cache.read_text()) if args.cache.exists() else {}

    scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    # Per-device clock, so readings from consecutive recordings never overlap in time.
    next_ts: dict[str, datetime] = {}
    # Round-robin over bearings, so a short replay still covers every held-out bearing.
    per_bearing = {b: sorted(g["recording_id"].unique()) for b, g in rows.groupby("bearing")}
    interleaved = [r for group in zip_longest(*per_bearing.values()) for r in group if r]
    recordings = interleaved[: args.max_recordings]
    step = timedelta(seconds=ws.X.shape[2] / ws.fs)
    for rec_id in recordings:
        idx = rows.index[rows["recording_id"] == rec_id].to_numpy()
        meta = ws.meta.iloc[idx[0]]
        name = f"{args.device_prefix}-{meta['source']}-{meta['bearing']}"
        key = device_key(platform, admin, name, cache)
        X = np.asarray(ws.X[idx])
        body = {
            "fs": ws.fs,
            "windows": [
                {"channels": {ch: X[i, c].tolist() for c, ch in enumerate(ws.channels)}}
                for i in range(len(X))
            ],
        }
        preds = model.post("/predict", json=body).raise_for_status().json()["predictions"]
        now = datetime.now(UTC)
        start = max(now - step * len(preds), next_ts.get(name, now - step * len(preds)))
        # The platform rejects timestamps more than 5 min ahead; let the clock catch up.
        ahead = (start + step * len(preds) - now).total_seconds() - 60
        if ahead > 0:
            time.sleep(ahead)
        next_ts[name] = start + step * len(preds)
        readings = [
            {
                "metric": METRIC,
                "value": p["fault_probability"],
                "ts": (start + step * i).isoformat(),
            }
            for i, p in enumerate(preds)
        ]
        platform.post(
            "/readings/batch",
            json={"readings": readings},
            headers={"X-API-Key": key, "Idempotency-Key": str(uuid.uuid4())},
        ).raise_for_status()
        scores[(meta["bearing"], meta["label"])].extend(p["fault_probability"] for p in preds)
        print(
            f"{rec_id}: true={meta['label']:<8} mean fault_prob="
            f"{np.mean([p['fault_probability'] for p in preds]):.2f}"
        )
        time.sleep(args.pause)
    args.cache.write_text(json.dumps(cache, indent=2))
    args.cache.chmod(0o600)
    return {f"{b} ({label})": float(np.mean(v)) for (b, label), v in sorted(scores.items())}


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m bearing.replay")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--held-out", type=Path, required=True)
    p.add_argument("--model-url", default="http://localhost:8100")
    p.add_argument("--platform-url", default="http://localhost:8000")
    p.add_argument("--admin-token", default="dev-admin-token")
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--max-recordings", type=int, default=40)
    p.add_argument("--pause", type=float, default=0.2, help="seconds between recordings")
    p.add_argument("--device-prefix", default="replay")
    p.add_argument("--cache", type=Path, default=Path(".replay_devices.json"))
    args = p.parse_args()
    with (
        httpx.Client(base_url=args.model_url, timeout=30) as model,
        httpx.Client(base_url=args.platform_url, timeout=30) as platform,
    ):
        summary = replay(args, model, platform)
    print("\nMean fault probability per held-out bearing:")
    for k, v in summary.items():
        print(f"  {k:<24} {v:.2f}")


if __name__ == "__main__":
    main()
