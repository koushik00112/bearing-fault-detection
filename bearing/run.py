"""Run the evaluation protocol: every scenario x model x seed, then write the report.

  python -m bearing.run --data data/processed/paderborn --out results/paderborn
  python -m bearing.run --data data/processed/synthetic --out results/synthetic --quick

For each seed, scenarios A/B/B2 draw a new split; C and D have fixed splits, so only the
model's randomness varies. Scenario C's score for a seed is the mean over its 4 folds.
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from bearing import CLASSES, drift, models
from bearing.evaluate import classification_metrics, latency_ms, model_size_bytes, noise_curve
from bearing.prepare import load
from bearing.report import write_report
from bearing.splits import SCENARIOS, build_splits, validation_split
from bearing.tracking import Tracker
from bearing.windowing import WindowSet

log = logging.getLogger("run")


@dataclass
class Config:
    scenarios: tuple[str, ...] = SCENARIOS
    models: tuple[str, ...] = ("rf", "hgb", "cnn")
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    test_size: float = 0.3
    val_fraction: float = 0.15
    noise_scenario: str = "B2_bearing"
    snrs_db: tuple[float, ...] = (20, 10, 5, 0, -5)
    export_scenario: str = "B2_bearing"
    healthy_test_bearings: tuple[str, ...] = ("K004", "K005", "K006")
    model_params: dict[str, dict[str, Any]] = field(default_factory=dict)


def class_order(labels: set[str]) -> tuple[str, ...]:
    if labels <= set(CLASSES):
        return tuple(c for c in CLASSES if c in labels)
    return tuple(sorted(labels))


def run(ws: WindowSet, cfg: Config, out: Path, tracker: Tracker) -> dict[str, Any]:
    classes = class_order(set(ws.meta["label"]))
    y_all = np.array([classes.index(label) for label in ws.meta["label"]])
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    efficiency: dict[str, dict[str, Any]] = {}

    for scenario in cfg.scenarios:
        for seed in cfg.seeds:
            splits = build_splits(ws.meta, scenario, seed, cfg.test_size, cfg.healthy_test_bearings)
            for split in splits:
                test = ws.subset(split.test)
                y_test = y_all[split.test]
                for name in cfg.models:
                    train_idx, val = split.train, None
                    if name == "cnn":
                        train_idx, val_idx = validation_split(
                            ws.meta, split.train, split.group_key, cfg.val_fraction, seed
                        )
                        val = ws.subset(val_idx)
                    train = ws.subset(train_idx)
                    model = models.make(name, classes, ws.fs, **cfg.model_params.get(name, {}))
                    params = {
                        "scenario": scenario,
                        "fold": split.fold,
                        "seed": seed,
                        "model": name,
                        "leaky": split.leaky,
                        "n_train": len(train),
                        "n_test": len(test),
                    }
                    with tracker.run(f"{scenario}/{split.fold}/{name}/s{seed}", params) as rec:
                        t0 = time.perf_counter()
                        model.fit(train, val, seed)
                        fit_s = time.perf_counter() - t0
                        proba = model.predict_proba(test.X)
                        m = classification_metrics(y_test, proba, classes)
                        rec["metrics"].update(
                            {
                                "macro_f1": m["macro_f1"],
                                "accuracy": m["accuracy"],
                                "fit_seconds": fit_s,
                            }
                        )
                    row = {
                        **params,
                        **m,
                        "fit_seconds": fit_s,
                        "test_bearings": sorted(test.meta["bearing"].unique()),
                    }
                    if scenario == cfg.noise_scenario:
                        row["noise_curve"] = noise_curve(model, test.X, y_test, cfg.snrs_db, seed)
                    rows.append(row)
                    log.info(
                        "%s/%s seed=%d %s macro_f1=%.3f (%.0fs)",
                        scenario,
                        split.fold,
                        seed,
                        name,
                        m["macro_f1"],
                        fit_s,
                    )

                    if seed == cfg.seeds[0]:
                        pred = proba.argmax(axis=1)
                        by = test.meta.assign(correct=pred == y_test)
                        for (bearing, cond), grp in by.groupby(["bearing", "condition"]):
                            errors.append(
                                {
                                    "scenario": scenario,
                                    "fold": split.fold,
                                    "model": name,
                                    "bearing": bearing,
                                    "condition": cond,
                                    "label": grp["label"].iloc[0],
                                    "accuracy": float(grp["correct"].mean()),
                                    "n": len(grp),
                                }
                            )
                    if seed == cfg.seeds[0] and scenario == cfg.export_scenario:
                        export = out / "models" / scenario / name
                        model.save(export)
                        held_out = sorted(test.meta["recording_id"].unique())
                        (export / "held_out_recordings.json").write_text(json.dumps(held_out))
                        (export / "serving.json").write_text(
                            json.dumps(
                                {
                                    "window": int(ws.X.shape[2]),
                                    "trained_on": {
                                        "scenario": scenario,
                                        "seed": seed,
                                        "dataset_source": str(ws.meta["source"].iloc[0]),
                                        "n_train_windows": len(train),
                                    },
                                    "drift_reference": drift.reference(train.X, ws.channels),
                                },
                                indent=2,
                            )
                        )
                        efficiency[name] = {
                            "latency_ms": latency_ms(model, test.X),
                            "size_bytes": model_size_bytes(model),
                        }
                        if hasattr(model, "n_parameters"):
                            efficiency[name]["n_parameters"] = model.n_parameters()

    return {"classes": list(classes), "rows": rows, "errors": errors, "efficiency": efficiency}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="python -m bearing.run")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scenarios", default=",".join(SCENARIOS))
    p.add_argument("--models", default="rf,hgb,cnn")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--quick", action="store_true", help="few CNN epochs, small forests (smoke)")
    p.add_argument("--no-mlflow", action="store_true")
    args = p.parse_args()

    ws = load(args.data)
    info = json.loads((args.data / "dataset.json").read_text())
    cfg = Config(
        scenarios=tuple(args.scenarios.split(",")),
        models=tuple(args.models.split(",")),
        seeds=tuple(int(s) for s in args.seeds.split(",")),
    )
    if "healthy_test_bearings" in info:
        cfg.healthy_test_bearings = tuple(info["healthy_test_bearings"])
    if args.quick:
        cfg.model_params = {
            "cnn": {"epochs": 3},
            "rf": {"n_estimators": 50},
            "hgb": {"max_iter": 50},
        }
    args.out.mkdir(parents=True, exist_ok=True)
    tracker = Tracker(args.out, experiment=args.out.name, use_mlflow=not args.no_mlflow)
    started = datetime.now(UTC)
    result = run(ws, cfg, args.out, tracker)
    result["run_info"] = {
        "started": started.isoformat(timespec="seconds"),
        "finished": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": info,
        "config": {**cfg.__dict__},
        "quick": args.quick,
    }
    (args.out / "results.json").write_text(json.dumps(result, indent=2, default=str))
    tracker.save()
    report = write_report(result, args.out)
    log.info("report written to %s", report)


if __name__ == "__main__":
    main()
