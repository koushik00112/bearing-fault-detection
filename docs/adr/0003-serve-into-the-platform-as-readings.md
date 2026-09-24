# ADR 0003: Integrate with the telemetry platform as ordinary readings

Date: 2026-09-24
Status: Accepted

## Context
The telemetry platform stores low-rate scalar readings and raises threshold alerts. The
model needs 16 kHz windows, which the platform isn't designed to store.

## Decision
- The model runs as its own service (`POST /predict` on raw windows).
- A bridge (`bearing.replay`, standing in for an edge gateway) sends windows to the model
  and posts the returned **fault probability** to the platform as the metric
  `bearing_fault_prob`, using the device's normal API key and batch endpoint with
  idempotency keys.
- Alerting reuses the platform's existing rule engine (`bearing_fault_prob gt 0.8`).

## Consequences
- No changes to the platform, and the model can be redeployed on its own.
- Raw waveforms aren't retained centrally. Re-scoring history with a new model needs the
  raw data from elsewhere (acceptable for a demo; a real system would archive windows to
  object storage).
- Demo data is replayed from held-out Paderborn recordings. Device names say `replay-...`
  so it can't be mistaken for live hardware.
