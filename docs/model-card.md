# Model card: bearing fault classifier

**Status:** filled from the run of 2026-09-25 (`results/paderborn/summary.md`, code
`d0eaaa3`, raw-data manifest `43bdbcc3…`).

> **Bottom line: not fit for use on bearings it wasn't trained on.** On unseen bearings
> (scenario B2), macro-F1 is about 0.5 for the best model, against about 0.33 for chance.

## Model details
- **Task:** classify a 128 ms window of vibration plus two motor-phase currents
  (16 kHz) as healthy, inner-race fault or outer-race fault.
- **Served model (demo):** random forest. It's best on B2 (0.513) and C (0.832). The
  gradient-boosting model is close; the CNN is weaker on B2 but about 65× faster (0.27 ms vs 17.9 ms).
  - Baselines: 22 engineered features per channel (time domain, spectral bands, envelope
    spectrum), then a random forest or histogram gradient boosting.
  - CNN: 4 conv blocks with a wide first kernel, global average pooling; 23,763 parameters.
- **Version:** reported by the service as `<model>-<sha256 prefix>` (`GET /model`).
- **Trained on:** scenario B2 seed 0 training split. The exact held-out recordings are
  listed in `held_out_recordings.json` next to the model.

## Intended use
- Portfolio and research demonstration of leakage-safe evaluation and model serving.
- Scoring streamed windows inside the telemetry platform, where a fault probability above
  a threshold raises an alert for **human review**.

## Out of scope
- Safety-critical or maintenance decisions without human review.
- Other bearing types, machines, sensors, mounting positions or sampling rates. All
  training data comes from one test rig with 6203 bearings.
- Commercial use: the training data is CC BY-NC 4.0.

## Training data
Paderborn KAt bearing dataset (see [data.md](data.md)), with 29 bearings (combined damage
excluded), 4 operating conditions, decimated to 16 kHz: 2,319 recordings (one file is
corrupt) and 37,104 windows. Raw data manifest sha256: `43bdbcc38a068740df4fe7393ecd0d66bb38e26c27de2b1b4a30124c9911a2d8`.

## Evaluation
Protocol: [evaluation-protocol.md](evaluation-protocol.md). Macro-F1, mean ± std over 5 seeds:

| Scenario | Random forest (served) | Notes |
|---|---|---|
| A (leaky, for comparison only) | 0.992 ± 0.000 | |
| B recording-level | 0.990 ± 0.002 | same bearings on both sides |
| B2 bearing-level | **0.513 ± 0.208** | headline; per seed 0.21 to 0.75 depending on which bearings are held out |
| C leave-one-condition-out | 0.832 ± 0.005 | worst fold: 900 rpm, 0.685 |
| D artificial → real | 0.412 ± 0.002 | |

Noise robustness (B2): 0.272 at 10 dB, 0.260 at 0 dB (the clean B2 score is already low).
Latency on an Apple M3 CPU, single window: p50 17.9 ms, p95 28.8 ms (mostly feature
extraction). Size: 27 MB on disk.

Why the gap between B and B2: the same features identify the *individual bearing* among 29
with 98.9% accuracy on held-out recordings (chance 3.4%), so a classifier trained with the
bearing in its training set can rely on the bearing's fingerprint rather than the fault.

## Limitations and known failure modes
- **Domain shift:** trained on artificial damage and tested on real damage (D), macro-F1 is
  0.41, against 0.99 when real-damage bearings are in training. Most test bearings are
  classified entirely one way; healthy K004 and K005 are always called faulty.
- **Unseen operating conditions** (C): fault frequencies scale with shaft speed; the
  900 rpm fold is the hardest. Measured: random forest 0.685, CNN 0.255.
- **Class set is closed:** combined damage, ball faults, misalignment or looseness will be
  forced into one of three classes.
- **Decimation** to 16 kHz removes the band above 8 kHz: 43% of vibration power on average
  in the first 160 real recordings, and very different between the two bearings measured
  (see ADR 0002). At 32 kHz, B2 improves by +0.04 on average (4 of 5 seeds for the
  random forest): small, not a fix.
- **Small number of physical bearings** (6 healthy, 11 inner, 12 outer), so B2 and D
  estimates rest on a handful of test bearings and have high variance. Read the ± values.

## Monitoring in production
- `input_drift_psi` per channel for RMS, kurtosis and crest factor, over the last 500
  windows vs training. Alert above 0.25.
- PSI also rises when the *fault mix* changes (for example, a fleet that is mostly
  healthy), so a drift alert means "look at the data", not "the model is broken".
- The platform alert rule on `bearing_fault_prob` (default above 0.8) is the operational
  output.

## Ethical and practical considerations
False negatives (a missed fault) risk equipment damage; false positives waste maintenance
effort and erode trust. The threshold trades these off and should be set from per-class
recall on B2 and D, with the people who act on the alerts.
