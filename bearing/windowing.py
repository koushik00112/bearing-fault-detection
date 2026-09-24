"""Cut recordings into fixed-length windows, keeping every window's provenance.

Every window row carries its recording_id, bearing and condition, so splits can be
made by recording (or bearing) and checked for leakage afterwards.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bearing.data.recording import Recording

META_COLUMNS = ["recording_id", "source", "bearing", "condition", "label", "origin", "start"]


@dataclass
class WindowSet:
    X: np.ndarray  # (n_windows, n_channels, window) float32 (may be a memmap)
    meta: pd.DataFrame  # one row per window, META_COLUMNS
    channels: tuple[str, ...]
    fs: float

    def __len__(self) -> int:
        return len(self.meta)

    def subset(self, idx: np.ndarray) -> "WindowSet":
        idx = np.asarray(idx)
        return WindowSet(
            X=np.asarray(self.X[idx], dtype=np.float32),
            meta=self.meta.iloc[idx].reset_index(drop=True),
            channels=self.channels,
            fs=self.fs,
        )


def window_starts(n_samples: int, window: int, stride: int, max_windows: int | None) -> list[int]:
    """Start indices of a contiguous block of windows from the middle of the recording.

    A contiguous block keeps neighbouring (overlapping) windows together, which is what makes
    the leaky random-window split leaky; taking it from the middle avoids start-up transients.
    """
    if n_samples < window:
        return []
    starts = list(range(0, n_samples - window + 1, stride))
    if max_windows is not None and len(starts) > max_windows:
        offset = (len(starts) - max_windows) // 2
        starts = starts[offset : offset + max_windows]
    return starts


def windows_of(
    rec: Recording, channels: tuple[str, ...], window: int, stride: int, max_windows: int | None
) -> tuple[np.ndarray, list[dict[str, object]]]:
    starts = window_starts(rec.n_samples, window, stride, max_windows)
    stacked = np.stack([rec.signals[c][: rec.n_samples] for c in channels])
    X = (
        np.stack([stacked[:, s : s + window] for s in starts])
        if starts
        else np.empty((0, len(channels), window), dtype=np.float32)
    )
    rows: list[dict[str, object]] = [
        {
            "recording_id": rec.recording_id,
            "source": rec.source,
            "bearing": rec.bearing,
            "condition": rec.condition,
            "label": rec.label,
            "origin": rec.origin,
            "start": s,
        }
        for s in starts
    ]
    return X.astype(np.float32), rows


def make_windows(
    recordings: Iterable[Recording],
    channels: tuple[str, ...],
    window: int,
    stride: int,
    max_windows: int | None = None,
) -> WindowSet:
    xs, rows, fs = [], [], None
    for rec in recordings:
        if fs is None:
            fs = rec.fs
        elif rec.fs != fs:
            raise ValueError(f"mixed sampling rates: {fs} and {rec.fs} ({rec.recording_id})")
        X, r = windows_of(rec, channels, window, stride, max_windows)
        xs.append(X)
        rows.extend(r)
    if fs is None:
        raise ValueError("no recordings")
    return WindowSet(
        X=np.concatenate(xs) if xs else np.empty((0, len(channels), window), np.float32),
        meta=pd.DataFrame(rows, columns=META_COLUMNS),
        channels=channels,
        fs=fs,
    )
