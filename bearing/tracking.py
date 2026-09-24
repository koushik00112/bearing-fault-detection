"""Experiment tracking: always a local JSON log; MLflow as well when it's installed.

The JSON log is the source of truth for the results table. MLflow is for browsing runs
(`mlflow ui --backend-store-uri results/mlruns`).
"""

import contextlib
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class Tracker:
    def __init__(self, out_dir: Path, experiment: str, use_mlflow: bool = True) -> None:
        self.out_dir = out_dir
        self.runs: list[dict[str, Any]] = []
        self.mlflow: Any = None
        if use_mlflow:
            try:
                import mlflow
            except ImportError:
                log.info("mlflow not installed; tracking to JSON only")
            else:
                mlflow.set_tracking_uri((out_dir / "mlruns").resolve().as_uri())
                mlflow.set_experiment(experiment)
                self.mlflow = mlflow

    @contextlib.contextmanager
    def run(self, name: str, params: dict[str, Any]) -> Iterator[dict[str, Any]]:
        record: dict[str, Any] = {"name": name, "params": params, "metrics": {}}
        ctx = self.mlflow.start_run(run_name=name) if self.mlflow else contextlib.nullcontext()
        with ctx:
            if self.mlflow:
                self.mlflow.log_params(params)
            yield record
            if self.mlflow:
                flat = {k: v for k, v in record["metrics"].items() if isinstance(v, int | float)}
                self.mlflow.log_metrics(flat)
        self.runs.append(record)

    def save(self) -> Path:
        path = self.out_dir / "runs.json"
        path.write_text(json.dumps(self.runs, indent=2))
        return path
