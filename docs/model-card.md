# Model card: bearing fault classifier

**Status: template.** Fill every "TBD" from `results/paderborn/summary.md` after the full
run. Don't estimate any of them.

## Model details
- **Task:** classify a 128 ms window of vibration plus two motor-phase currents
  (16 kHz) as healthy, inner-race fault or outer-race fault.
- **Served model:** TBD (rf / hgb / cnn). Choose on scenario B2 and D performance, not A.
  - Baselines: 22 engineered features per channel (time domain, spectral bands, envelope
    spectrum), then a random forest or histogram gradient boosting.
  - CNN: 4 conv blocks with a wide first kernel, global average pooling; TBD parameters.
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
excluded), 4 operating conditions, decimated to 16 kHz. Raw data manifest hash: TBD.

## Evaluation
Protocol: [evaluation-protocol.md](evaluation-protocol.md). Macro-F1, mean ± std over 5 seeds:

| Scenario | Served model | Notes |
|---|---|---|
| A (leaky, for comparison only) | TBD | |
| B recording-level | TBD | |
| B2 bearing-level | TBD | headline |
| C leave-one-condition-out | TBD | worst fold: TBD |
| D artificial → real | TBD | |

Noise robustness (B2): TBD at 10 dB, TBD at 0 dB.
Latency p50/p95 on TBD hardware: TBD ms. Size: TBD.

## Limitations and known failure modes
- **Domain shift:** performance on real damage when trained on artificial damage (D) is
  expected to be worse. Quote the measured gap: TBD.
- **Unseen operating conditions** (C): fault frequencies scale with shaft speed; the
  900 rpm fold is the likeliest to fail. Measured: TBD.
- **Class set is closed:** combined damage, ball faults, misalignment or looseness will be
  forced into one of three classes.
- **Decimation** to 16 kHz removes the band above 8 kHz: 43% of vibration power on average
  in the first 160 real recordings, and very different between the two bearings measured
  (see ADR 0002). The 32 kHz ablation result: TBD.
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
