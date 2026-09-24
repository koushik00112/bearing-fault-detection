# ADR 0001: Add a bearing-level split (B2) and make it the headline number

Date: 2026-09-24
Status: Accepted

## Context
The plan specified four scenarios: A leaky, B recording-level, C leave-one-condition-out,
and D artificial to real. In Paderborn, each physical bearing contributes 80 recordings (4
conditions × 20). A recording-level split (B) keeps recordings intact, but the same
physical bearing still sits on both sides, so a model can learn that bearing's individual
signature.

## Decision
Add **B2: bearing-level split** (per class, 30% of bearings to test, at least one), keep B,
and treat B2 as the headline number. Reporting both makes the bearing-identity effect
measurable (B minus B2).

## Consequences
- With 6 healthy, 11 inner-race and 12 outer-race bearings, B2's test set has only 2, 3 and 4
  bearings per class respectively, so the spread over seeds will be large. That's the honest uncertainty, and the report shows ± values.
- More compute (one extra scenario).
