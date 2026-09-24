# Bearing Fault Detection

Classifies bearing faults (healthy / inner race / outer race) from vibration and motor
current, with an evaluation protocol designed to show **how much of the score survives
when the test data is genuinely new**. The model is served as an API and plugged into the
[telemetry platform](../telemetry-platform), where high fault probabilities raise alerts.

- **Main data:** Paderborn University KAt bearing dataset (real test-rig recordings,
  CC BY-NC 4.0). **CWRU** is used only as a sanity check. **Synthetic** signals are used
  only for tests and demos, and are labelled as such everywhere.
- **Evaluation:** four scenarios, from deliberately leaky to artificial-to-real damage,
  with a leakage guard enforced in code and CI. See
  [docs/evaluation-protocol.md](docs/evaluation-protocol.md).

[Protocol](docs/evaluation-protocol.md) · [Data and licences](docs/data.md) ·
[Model card](docs/model-card.md) · [Decisions](docs/adr)

## Results

**Not run yet.** The table below is filled from `results/paderborn/summary.md` after the
full run on Paderborn data. Nothing here is estimated.

| Scenario | RF | HGB | 1D CNN |
|---|---|---|---|
| A: random windows (leaky, on purpose) | | | |
| B: recording-level | | | |
| B2: bearing-level (headline) | | | |
| C: leave one condition out | | | |
| D: artificial → real damage | | | |

Macro-F1, mean ± std over 5 seeds. Hypothesis (stated in advance, from published
findings): A is highest; B2, C and D drop. The results will be reported whatever they show.

## Quick start (no download, synthetic data)

```bash
make install
make test           # 109 tests, including the leakage guard
make smoke          # the whole protocol on synthetic data -> results/synthetic-smoke/summary.md
```

The smoke report opens with a banner saying its numbers are synthetic and meaningless.

## Real data

```bash
make download-sample   # K001 + KA01 (~330 MB); check the loader against real files, see docs/data.md
make download          # remaining 3-class bearings (~4.6 GB) + CWRU; asks before downloading
make prepare           # 16 kHz, 2048-sample windows -> data/processed/paderborn (~0.9 GB)
make results           # 5 scenarios x 3 models x 5 seeds -> results/paderborn/summary.md
```

Raw data, processed windows and trained models are licensed derivatives and are never
committed (`.gitignore`). Only aggregate results are committed.

## Serving and platform integration

```bash
# 1. Serve a model exported by the run (trained on scenario B2, seed 0)
cp -r results/paderborn/models/B2_bearing/hgb models/serving
docker compose up --build            # model API on :8100 (WITH_TORCH=1 for the CNN)

# 2. Start the telemetry platform (other repo) on :8000, then replay held-out recordings
python -m bearing.replay --data data/processed/paderborn \
  --held-out models/serving/held_out_recordings.json
```

The replay sends windows from recordings the model never saw to `POST /predict` and posts
`bearing_fault_prob` to the platform. The platform's alert worker then opens an alert when
it exceeds 0.8. Checked end to end locally with synthetic data (2026-09-24): the held-out
healthy bearing stayed at 0.00 with no alert; the inner- and outer-race bearings scored
1.00 and each raised an alert. That verifies the plumbing, not model quality.

The model service exposes `/predict`, `/model`, `/drift` (PSI against training inputs),
`/readyz`, `/livez` and Prometheus `/metrics`.

## Layout

```
bearing/data/        loaders (paderborn, cwru, synthetic) and download
bearing/windowing.py windows with provenance (recording, bearing, condition)
bearing/splits.py    the scenarios and the leakage guard
bearing/features.py  engineered features;  bearing/models/  rf, hgb, cnn
bearing/run.py       protocol runner;  bearing/report.py  summary.md and figures
bearing/serve/       FastAPI model service;  bearing/drift.py  PSI monitoring
bearing/replay.py    bridge into the telemetry platform
```

## Citation

If you use the Paderborn data: Lessmeier et al., KAt-DataCenter, Chair of Design and Drive
Technology, Paderborn University; and Lessmeier, Kimotho, Zimmer, Sextro (2016), PHM
Society European Conference. See [docs/data.md](docs/data.md).
