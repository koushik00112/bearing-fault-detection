"""The leakage guard and the scenario splits. These tests are the evaluation protocol's safety net."""

import numpy as np
import pytest

from bearing.data.synthetic import HEALTHY_TEST_BEARINGS
from bearing.splits import (
    SCENARIOS,
    LeakageError,
    Split,
    assert_disjoint,
    build_splits,
    validation_split,
)


def test_guard_raises_when_a_recording_is_on_both_sides(ws):
    rec = ws.meta["recording_id"].iloc[0]
    rows = np.flatnonzero(ws.meta["recording_id"] == rec)
    others = np.flatnonzero(ws.meta["recording_id"] != rec)
    leaky = Split(
        "test", "x", train=np.r_[rows[:1], others[:10]], test=rows[1:], group_key="recording_id"
    )
    with pytest.raises(LeakageError, match=rec):
        assert_disjoint(ws.meta, leaky)


def test_guard_passes_a_clean_split(ws):
    ids = ws.meta["recording_id"]
    first = ids.unique()[:5]
    split = Split(
        "t", "x", np.flatnonzero(~ids.isin(first)), np.flatnonzero(ids.isin(first)), "recording_id"
    )
    assert_disjoint(ws.meta, split)


def test_guard_rejects_empty_sides(ws):
    with pytest.raises(ValueError, match="empty"):
        assert_disjoint(ws.meta, Split("t", "x", np.arange(10), np.array([], int), "recording_id"))


@pytest.mark.parametrize("scenario", [s for s in SCENARIOS if s != "A_leaky_window"])
@pytest.mark.parametrize("seed", range(10))
def test_no_recording_is_ever_on_both_sides(ws, scenario, seed):
    for split in build_splits(ws.meta, scenario, seed, healthy_test_bearings=HEALTHY_TEST_BEARINGS):
        train = set(ws.meta["recording_id"].iloc[split.train])
        test = set(ws.meta["recording_id"].iloc[split.test])
        assert not train & test
        assert len(split.train) + len(split.test) <= len(ws.meta)
        assert not np.intersect1d(split.train, split.test).size


def test_leaky_split_really_is_leaky(ws):
    # If this ever fails, scenario A no longer measures what it claims to.
    [split] = build_splits(ws.meta, "A_leaky_window", seed=0)
    assert split.leaky
    shared = set(ws.meta["recording_id"].iloc[split.train]) & set(
        ws.meta["recording_id"].iloc[split.test]
    )
    assert len(shared) > 0.5 * ws.meta["recording_id"].nunique()


@pytest.mark.parametrize("seed", range(5))
def test_bearing_split_keeps_physical_bearings_apart_and_covers_every_class(ws, seed):
    [split] = build_splits(ws.meta, "B2_bearing", seed)
    train_b = set(ws.meta["bearing"].iloc[split.train])
    test_b = set(ws.meta["bearing"].iloc[split.test])
    assert not train_b & test_b
    assert set(ws.meta["label"].iloc[split.test]) == {"healthy", "inner", "outer"}
    assert set(ws.meta["label"].iloc[split.train]) == {"healthy", "inner", "outer"}


def test_recording_split_puts_every_bearing_condition_on_both_sides(ws):
    [split] = build_splits(ws.meta, "B_recording", seed=3)
    tr = ws.meta.iloc[split.train].groupby(["bearing", "condition"]).size()
    te = ws.meta.iloc[split.test].groupby(["bearing", "condition"]).size()
    assert set(tr.index) == set(te.index)


def test_loco_has_one_fold_per_condition_and_tests_only_that_condition(ws):
    splits = build_splits(ws.meta, "C_loco", seed=0)
    assert sorted(s.fold for s in splits) == sorted(ws.meta["condition"].unique())
    for s in splits:
        assert set(ws.meta["condition"].iloc[s.test]) == {s.fold}
        assert s.fold not in set(ws.meta["condition"].iloc[s.train])


def test_artificial_to_real_never_trains_on_real_damage(ws):
    [s] = build_splits(
        ws.meta, "D_artificial_to_real", 0, healthy_test_bearings=HEALTHY_TEST_BEARINGS
    )
    train, test = ws.meta.iloc[s.train], ws.meta.iloc[s.test]
    assert set(train["origin"]) <= {"artificial", "none"}
    assert set(test["origin"]) <= {"real", "none"}
    present = set(HEALTHY_TEST_BEARINGS) & set(ws.meta["bearing"])
    assert present and set(test.loc[test["label"] == "healthy", "bearing"]) == present
    assert not set(train["bearing"]) & set(test["bearing"])


@pytest.mark.parametrize("group_key", ["recording_id", "bearing"])
def test_validation_split_is_grouped_by_recording(ws, group_key):
    [split] = build_splits(ws.meta, "B_recording", seed=0)
    tr, va = validation_split(ws.meta, split.train, group_key, 0.2, seed=0)
    assert len(tr) and len(va)
    assert not set(ws.meta["recording_id"].iloc[tr]) & set(ws.meta["recording_id"].iloc[va])
    assert set(tr) | set(va) == set(split.train)


def test_splits_are_deterministic_per_seed(ws):
    a = build_splits(ws.meta, "B2_bearing", seed=7)[0]
    b = build_splits(ws.meta, "B2_bearing", seed=7)[0]
    assert np.array_equal(a.test, b.test)
    tests = {tuple(build_splits(ws.meta, "B2_bearing", seed=s)[0].test) for s in range(10)}
    assert len(tests) > 1  # seeds actually change the split


def test_unknown_scenario():
    import pandas as pd

    with pytest.raises(ValueError, match="unknown scenario"):
        build_splits(pd.DataFrame(), "Z", 0)
