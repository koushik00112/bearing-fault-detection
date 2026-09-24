"""Train/test splits for the four evaluation scenarios, plus the leakage guard.

A  leaky_window         Random windows. Overlapping windows of one recording land on both
                        sides. Deliberately leaky, to measure how much it inflates scores.
B  recording            Whole recordings go to one side. The same physical bearing still
                        appears on both sides (in different recordings).
B2 bearing              Whole bearings go to one side. Stricter than B; added because B
                        still lets a model memorise bearing-specific signatures.
C  loco                 Leave one operating condition out (one fold per condition).
D  artificial_to_real   Train on artificially damaged bearings, test on real damage.
                        Healthy bearings are split by bearing between the two sides.

Every non-leaky split is checked with `assert_disjoint` before any model sees it.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


class LeakageError(AssertionError):
    """A recording (or bearing) appears in both train and test."""


@dataclass(frozen=True)
class Split:
    scenario: str
    fold: str
    train: np.ndarray  # row indices into the window meta table
    test: np.ndarray
    group_key: str  # the unit kept on one side: "window" | "recording_id" | "bearing"
    leaky: bool = False


def assert_disjoint(meta: pd.DataFrame, split: Split, key: str = "recording_id") -> None:
    """Raise LeakageError if any value of `key` occurs in both train and test."""
    overlap = set(meta[key].iloc[split.train]) & set(meta[key].iloc[split.test])
    if overlap:
        sample = sorted(overlap)[:5]
        raise LeakageError(
            f"{split.scenario}/{split.fold}: {len(overlap)} {key} value(s) in both train and "
            f"test, e.g. {sample}"
        )
    if len(split.train) == 0 or len(split.test) == 0:
        raise ValueError(f"{split.scenario}/{split.fold}: empty train or test set")


def _pick(groups: list[str], fraction: float, rng: np.random.Generator) -> set[str]:
    groups = sorted(groups)
    n = max(1, round(len(groups) * fraction)) if len(groups) > 1 else 0
    return set(rng.choice(groups, size=n, replace=False)) if n else set()


def leaky_window_split(meta: pd.DataFrame, test_size: float, seed: int) -> Split:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(meta))
    n_test = round(len(meta) * test_size)
    return Split(
        "A_leaky_window",
        "all",
        np.sort(idx[n_test:]),
        np.sort(idx[:n_test]),
        group_key="window",
        leaky=True,
    )


def recording_split(meta: pd.DataFrame, test_size: float, seed: int) -> Split:
    """Per (bearing, condition), send a fraction of its recordings to test."""
    rng = np.random.default_rng(seed)
    test_recs: set[str] = set()
    for _, grp in meta.groupby(["bearing", "condition"], sort=True):
        test_recs |= _pick(list(grp["recording_id"].unique()), test_size, rng)
    is_test = meta["recording_id"].isin(test_recs).to_numpy()
    return Split(
        "B_recording",
        "all",
        np.flatnonzero(~is_test),
        np.flatnonzero(is_test),
        group_key="recording_id",
    )


def bearing_split(meta: pd.DataFrame, test_size: float, seed: int) -> Split:
    """Per class, send a fraction of its physical bearings to test (at least one)."""
    rng = np.random.default_rng(seed)
    test_bearings: set[str] = set()
    for _, grp in meta.groupby("label", sort=True):
        test_bearings |= _pick(list(grp["bearing"].unique()), test_size, rng)
    is_test = meta["bearing"].isin(test_bearings).to_numpy()
    return Split(
        "B2_bearing", "all", np.flatnonzero(~is_test), np.flatnonzero(is_test), group_key="bearing"
    )


def leave_one_condition_out(meta: pd.DataFrame) -> list[Split]:
    splits = []
    for cond in sorted(meta["condition"].unique()):
        is_test = (meta["condition"] == cond).to_numpy()
        splits.append(
            Split(
                "C_loco",
                cond,
                np.flatnonzero(~is_test),
                np.flatnonzero(is_test),
                group_key="recording_id",
            )
        )
    return splits


def artificial_to_real(meta: pd.DataFrame, healthy_test_bearings: tuple[str, ...]) -> Split:
    healthy = meta["label"] == "healthy"
    held_out_healthy = healthy & meta["bearing"].isin(healthy_test_bearings)
    train = (meta["origin"] == "artificial") | (healthy & ~held_out_healthy)
    test = (meta["origin"] == "real") | held_out_healthy
    return Split(
        "D_artificial_to_real",
        "all",
        np.flatnonzero(train.to_numpy()),
        np.flatnonzero(test.to_numpy()),
        group_key="bearing",
    )


def build_splits(
    meta: pd.DataFrame,
    scenario: str,
    seed: int,
    test_size: float = 0.3,
    healthy_test_bearings: tuple[str, ...] = ("K004", "K005", "K006"),
) -> list[Split]:
    """All folds for one scenario, each already checked for leakage (except A)."""
    if scenario == "A_leaky_window":
        splits = [leaky_window_split(meta, test_size, seed)]
    elif scenario == "B_recording":
        splits = [recording_split(meta, test_size, seed)]
    elif scenario == "B2_bearing":
        splits = [bearing_split(meta, test_size, seed)]
    elif scenario == "C_loco":
        splits = leave_one_condition_out(meta)
    elif scenario == "D_artificial_to_real":
        splits = [artificial_to_real(meta, healthy_test_bearings)]
    else:
        raise ValueError(f"unknown scenario {scenario}")
    for s in splits:
        if not s.leaky:
            assert_disjoint(meta, s, "recording_id")
            if s.group_key == "bearing":
                assert_disjoint(meta, s, "bearing")
    return splits


def validation_split(
    meta: pd.DataFrame, train_idx: np.ndarray, group_key: str, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Carve a validation set out of the training rows, grouped the same way as the scenario.

    Early stopping on a leaky validation set would quietly reintroduce the leak.
    """
    rng = np.random.default_rng(seed + 10_007)
    sub = meta.iloc[train_idx]
    if group_key == "window":
        perm = rng.permutation(len(train_idx))
        n_val = max(1, round(len(train_idx) * fraction))
        return np.sort(train_idx[perm[n_val:]]), np.sort(train_idx[perm[:n_val]])
    # Group by recording (always at least as strict as needed inside the training set).
    val_groups: set[str] = set()
    for _, grp in sub.groupby("label", sort=True):
        val_groups |= _pick(list(grp["recording_id"].unique()), fraction, rng)
    is_val = sub["recording_id"].isin(val_groups).to_numpy()
    return train_idx[~is_val], train_idx[is_val]


SCENARIOS = ("A_leaky_window", "B_recording", "B2_bearing", "C_loco", "D_artificial_to_real")
