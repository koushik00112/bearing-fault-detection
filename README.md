# Bearing Fault Detection

[![CI](https://github.com/koushik00112/bearing-fault-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/koushik00112/bearing-fault-detection/actions/workflows/ci.yml)

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

## Results (measured 2026-09-25, Paderborn, 2,319 recordings, 16 kHz)

**Headline: the models score 0.99 when the test bearings were seen in training, and
0.34–0.51 when they weren't.** For three classes, chance is about 0.33.

| Scenario | Random forest | Gradient boosting | 1D CNN |
|---|---|---|---|
| A: random windows (leaky, on purpose) | 0.992 ± 0.000 | 0.997 ± 0.001 | 0.964 ± 0.008 |
| B: new recordings of seen bearings | 0.990 ± 0.002 | 0.995 ± 0.001 | 0.950 ± 0.016 |
| **B2: bearings never seen in training** | **0.513 ± 0.208** | **0.495 ± 0.236** | **0.339 ± 0.142** |
| C: operating condition never seen | 0.832 ± 0.005 | 0.779 ± 0.000 | 0.561 ± 0.024 |
| D: train on artificial damage, test on real | 0.412 ± 0.002 | 0.429 ± 0.000 | 0.451 ± 0.012 |

Macro-F1, mean ± sample std over 5 seeds. Gradient boosting's ±0.000 in C and D isn't
stability: the model is deterministic, and those splits don't change with the seed. Full
report with per-class recall, confusion matrices, per-fold C, noise and error analysis:
[results/paderborn/summary.md](results/paderborn/summary.md).

**Why A and B look so good:** each physical bearing has a strong individual fingerprint.
Given a window from a recording it has never seen, a random forest names the exact bearing
out of 29 with **98.9%** accuracy, where chance is 3.4% (`results/paderborn/extra_checks.json`).
When those bearings are also in training (A, B), the classifier can lean on that fingerprint.
Hold the bearings out (B2) and most of the score disappears. In D, most test bearings are
classified *entirely* one way (per-bearing accuracy of 0.00 or 1.00). For example, healthy
bearings K004 and K005 are always called faulty, while K006 is always right.

**Other findings:**
- **Unseen conditions (C):** the 900 rpm fold is hardest (random forest 0.69, CNN 0.26),
  because fault frequencies scale with shaft speed. The 1500 rpm, 0.7 Nm, 1000 N fold is
  easiest (tree models 0.95–0.98, CNN 0.85).
- **32 kHz instead of 16 kHz** (B and B2 only, [ADR 0002](docs/adr/0002-preprocessing.md)):
  B2 improves slightly, by +0.04 on average, and random forest and gradient boosting
  improve in 4 of 5 seeds. That's real but small next to the ±0.2 spread, and it doesn't
  change the conclusion. B is unchanged.
- **CWRU sanity check:** 0.91–0.98 even with the load held out. It's much easier than
  Paderborn, which is why it's only a sanity check.
- **Noise:** on B2, the tree models drop to about 0.25–0.31 from 20 dB SNR downwards; the
  CNN holds until about 5 dB. With a clean B2 score this low, the noise curve says little.
- **Latency** (single window, Apple M3 CPU): CNN 0.27 ms, 103 KiB. Gradient boosting
  13.6 ms and random forest 17.9 ms, dominated by feature extraction.

**Hypothesis check:** stated in advance, "A highest; B2, C and D drop." That held, but the
size of the B2 drop (0.99 to about 0.5) was larger than expected, and B was *not* lower
than A: recording-level splitting alone removes almost none of the inflation on this
dataset.

**What this means:** these models don't reliably detect faults on bearings they haven't
seen. Bearing-level evaluation is essential, and a random-window or even recording-level
split would have reported about 0.99 for a model that is close to guessing on new
hardware. Improving B2 and D (domain adaptation, per-bearing normalisation, features
invariant to speed) is the natural next step.

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
