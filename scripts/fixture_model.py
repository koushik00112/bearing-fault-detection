"""Train a tiny model on SYNTHETIC data and export it for serving tests (CI, Docker smoke).

  python scripts/fixture_model.py models/fixture

Never use this model for anything but testing the serving path.
"""

import json
import sys
from pathlib import Path

from bearing import drift, models
from bearing.data import synthetic
from bearing.data.paderborn import CHANNELS
from bearing.windowing import make_windows


def main(out: Path) -> None:
    recs = synthetic.make_dataset(bearings_per_class=2, reps=2, seconds=0.5, seed=0)
    ws = make_windows(recs, CHANNELS, window=2048, stride=1024)
    model = models.make("rf", ("healthy", "inner", "outer"), ws.fs, n_estimators=20)
    model.fit(ws, None, seed=0)
    model.save(out)
    (out / "serving.json").write_text(
        json.dumps(
            {
                "window": 2048,
                "trained_on": {"dataset_source": "synthetic", "note": "FIXTURE MODEL, tests only"},
                "drift_reference": drift.reference(ws.X, ws.channels),
            },
            indent=2,
        )
    )
    print(f"fixture model written to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "models/fixture"))
