# ADR 0002: Decimate to 16 kHz by default, and measure what that costs

Date: 2026-09-24
Status: Accepted; revised the same day after measuring real files

## Context
At the raw 64 kHz, the full 3-class set (2,320 recordings × 4 s × 3 channels) is
about 7 GB as float32, too much to train on comfortably on a laptop, and the CNN would need
very long windows to cover a few shaft revolutions.

## Measurement (2026-09-24, K001 + KA01, 160 real recordings)
Share of vibration power above a cut-off, mean over 20 recordings per bearing and condition:

| Bearing | > 8 kHz | > 16 kHz |
|---|---|---|
| K001 (healthy), 4 conditions | 63–72% | 40–48% |
| KA01 (artificial outer-race damage), 4 conditions | 11–32% | 2–10% |
| All 160 recordings | 42.9% (range 11–73%) | 23.3% |

Motor currents are unaffected: their RMS is unchanged by decimation, and their
fundamental is 100 Hz at 1500 rpm and 60 Hz at 900 rpm.

**What this means:** decimating to 16 kHz discards a large share of the vibration signal,
and that share differs strongly *between bearings*. With two bearings, it's impossible to
say whether the difference is caused by the fault or by the individual bearing and
mounting. If it's the latter, a model using high frequencies would be learning bearing
identity, which is exactly what scenario B2 is designed to expose.

## Decision
- Default: zero-phase FIR decimation by 4 to **16 kHz**, 2048-sample windows (128 ms,
  about 3 revolutions at 1500 rpm), stride 1024, a contiguous middle block of 16 windows
  per recording (about 0.9 GB).
- **Required ablation:** repeat scenarios B and B2 at **32 kHz** (`--decimate 2 --window
  4096 --stride 2048`, the same 128 ms, about 1.8 GB) and report both. Limited to B and B2
  because those answer the question below, and all five scenarios would add about 4 h of
  CNN training (estimated from timing on the sample: about 1 s per epoch per 2,560 windows
  at 16 kHz). The interesting number is
  whether 32 kHz helps in B and B2 or only in B, where the bearing-identity shortcut is
  available.

## Consequences
- The 50% overlap is what makes scenario A leaky. Keeping it is deliberate: it's the common
  practice being measured.
- All parameters are CLI flags on `bearing.prepare` and recorded in `dataset.json`.
