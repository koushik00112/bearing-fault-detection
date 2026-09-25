# Evaluation protocol

The question this project answers isn't "how high can the score go?" It is **"how much of
a bearing-fault classifier's score survives when the test data is genuinely new?"**

## The leakage problem

A 4-second recording is cut into overlapping windows. Windows from the same recording
share most of their samples, the same bearing and the same sensor mounting. If windows are
shuffled into train and test at random, the model is tested on near-copies of its training
data, and the score measures memorisation, not diagnosis. This is a known pitfall in the
bearing-diagnosis literature (see the CWRU references in [data.md](data.md)).

## Scenarios

| | Split | Unit kept on one side | What it measures |
|---|---|---|---|
| **A** | Random windows, 70/30 | nothing (**leaky on purpose**) | The inflated number many tutorials report |
| **B** | Recording-level: per (bearing, condition), 30% of recordings go to test | recording | New recordings of bearings already seen in training |
| **B2** | Bearing-level: per class, 30% of physical bearings go to test | bearing | New bearings. The main "honest" number |
| **C** | Leave one operating condition out (4 folds) | recording (condition) | New speed, torque or load |
| **D** | Train on artificial damage plus K001–K003; test on real damage plus K004–K006 | bearing | Lab-made damage generalising to real wear |

B2 isn't in the original plan. It was added because B still lets the same *physical*
bearing appear on both sides, so a model could learn bearing-specific signatures (mounting,
manufacturing) instead of fault signatures. Comparing B with B2 measures that effect.

Scenario D mirrors the setup Lessmeier et al. (2016) used to show the domain shift between
artificial and real damage.

## Guarantees enforced in code
- `splits.build_splits` calls `assert_disjoint` on every non-leaky split: no
  `recording_id` on both sides, and no `bearing` on both sides for B2 and D. A violation
  raises `LeakageError` before any model trains.
- `tests/test_splits.py` checks disjointness for every scenario over 10 seeds, checks that
  scenario A really is leaky (so it keeps measuring what it claims), and runs as its own
  CI step.
- The CNN's early-stopping validation set is split by recording inside the training set,
  because a leaky validation set would quietly reintroduce the leak.
- Normalisation (the CNN's channel statistics, the baselines' scaler) is fitted on training
  data only, inside the model object, and the serving path uses the same code.
- Noise is added to test windows only.

## Metrics
- **Macro-F1** over the classes present in the test set (the headline), with **per-class
  recall** and **confusion matrices**.
- **5 seeds**, reported as mean ± sample standard deviation. A, B and B2 draw a new split
  per seed; C and D have fixed splits, so only model randomness varies. For C, each seed's
  score is the mean of its 4 folds, and the per-fold table is reported separately.
- **Noise robustness:** white Gaussian noise at 20, 10, 5, 0 and −5 dB SNR on B2's test
  windows.
- **Latency:** single-window inference including feature extraction, p50/p95, on the
  machine the report states. **Size:** serialized model bytes, plus the CNN's parameter
  count.

## Hypothesis (stated before running; not a result)
A scores highest; B is close to A; B2, C and D drop, with D likely the largest drop. This
comes from published findings and the reasoning above. The report shows whatever the data
says, including if the hypothesis is wrong.

## Error analysis
`error_analysis.csv` gives accuracy per test bearing and condition for the first seed, and
the report lists the 15 worst. Write-up questions to answer from it: Which bearings fail,
and are they the same across models? Does performance collapse at 900 rpm (scenario C,
where fault frequencies shift)? In D, which real-damage bearings get confused, and with
which class?

## Outcome (run of 2026-09-25)
The hypothesis held, but the size of the effect was the surprise: from 0.99 (A and B) to
about 0.5 (B2) for the best models. B was *not* meaningfully below A, so on Paderborn
splitting by recording removes almost none of the inflation; splitting by bearing is what
matters. A supporting check found that the features identify the individual bearing among
29 with 98.9% accuracy on held-out recordings. Numbers and discussion: the README's
Results section and `results/paderborn/summary.md`.
