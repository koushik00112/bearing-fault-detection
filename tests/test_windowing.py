import numpy as np
import pytest

from bearing.data import synthetic
from bearing.windowing import make_windows, window_starts


def test_window_starts_take_a_contiguous_middle_block():
    starts = window_starts(n_samples=20_000, window=2048, stride=1024, max_windows=4)
    assert np.all(np.diff(starts) == 1024)
    all_starts = list(range(0, 20_000 - 2048 + 1, 1024))
    offset = (len(all_starts) - 4) // 2
    assert starts == all_starts[offset : offset + 4]


def test_window_starts_short_recording():
    assert window_starts(1000, 2048, 1024, None) == []


def test_every_window_knows_its_recording(recordings):
    ws = make_windows(recordings[:3], ("vibration_1",), 2048, 1024)
    assert len(ws) == len(ws.meta) == ws.X.shape[0]
    assert set(ws.meta["recording_id"]) == {r.recording_id for r in recordings[:3]}
    rec = recordings[0]
    row = ws.meta.iloc[1]
    np.testing.assert_array_equal(
        ws.X[1, 0], rec.signals["vibration_1"][row.start : row.start + 2048]
    )


def test_mixed_sampling_rates_are_rejected():
    a = synthetic.make_recording("healthy", "SH00", "c1500", 0, 1, fs=16_000)
    b = synthetic.make_recording("healthy", "SH00", "c1500", 1, 2, fs=12_000)
    with pytest.raises(ValueError, match="mixed sampling rates"):
        make_windows([a, b], ("vibration_1",), 2048, 1024)


def test_subset_materialises_rows(ws):
    sub = ws.subset(np.array([0, 5, 9]))
    assert sub.X.shape[0] == 3 and len(sub.meta) == 3
    np.testing.assert_array_equal(sub.X[1], ws.X[5])
