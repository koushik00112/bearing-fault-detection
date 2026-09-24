# ADR 0002: Decimate to 16 kHz and take 16 windows per recording

Date: 2026-09-24
Status: Accepted, revisit if results suggest high-frequency content matters

## Context
At the raw 64 kHz, the full 3-class set (2,320 recordings × 4 s × 3 channels) is
about 7 GB as float32, too much to train on comfortably on a laptop, and the CNN would need
very long windows to cover a few shaft revolutions.

## Decision
- Zero-phase FIR decimation by 4 gives 16 kHz (content up to 8 kHz kept).
- 2048-sample windows (128 ms, about 3 revolutions at 1500 rpm) with a stride of 1024
  (50% overlap).
- A contiguous middle block of 16 windows per recording, about 0.9 GB total, stored as a
  memory-mapped file.

## Consequences
- Bearing resonances above 8 kHz are lost. Early-stage faults can show up there first, so
  this may understate what's achievable. It's noted in the model card.
- The 50% overlap is what makes scenario A leaky. Keeping it is deliberate: it's the common
  practice being measured.
- All parameters are CLI flags on `bearing.prepare`, and `dataset.json` records them.
