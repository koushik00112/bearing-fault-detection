"""Loaders are tested against files built to the documented layout.

Paderborn: verified against KAt's documented structure (struct with field Y, entries with
Name/Data). Re-check with one real file after downloading; see docs/data.md.
"""

import numpy as np
import pytest
import scipy.io

from bearing.data import cwru, paderborn


def write_paderborn_mat(path, channels, n=64_000 * 4 // 100):
    names = [*channels, "speed", "torque"]
    Y = np.empty(
        (1, len(names)),
        dtype=[("Name", "O"), ("Type", "O"), ("Unit", "O"), ("Data", "O"), ("Raster", "O")],
    )
    for i, name in enumerate(names):
        Y[0, i] = (
            name,
            "",
            "",
            np.sin(np.arange(n + i) / (5 + i)).astype(np.float64)[None, :],
            "HostService",
        )
    scipy.io.savemat(path, {path.stem: {"Info": "x", "X": np.zeros(1), "Y": Y, "Description": "d"}})


def test_bearing_table_matches_the_paper():
    origins = [origin for _label, origin in paderborn.BEARINGS.values()]
    assert len(paderborn.BEARINGS) == 32
    assert origins.count("none") == 6
    assert origins.count("artificial") == 12
    assert origins.count("real") == 14


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("N15_M07_F10_KA01_3.mat", ("N15_M07_F10", "KA01", 3)),
        ("N09_M07_F10_K001_20.mat", ("N09_M07_F10", "K001", 20)),
        ("N15_M01_F10_KI14_1.mat", ("N15_M01_F10", "KI14", 1)),
    ],
)
def test_parse_filename(name, expected):
    assert paderborn.parse_filename(name) == expected


def test_parse_filename_rejects_other_files():
    with pytest.raises(ValueError):
        paderborn.parse_filename("readme.mat")


def test_loads_documented_layout_and_decimates(tmp_path):
    path = tmp_path / "N15_M07_F10_KA01_1.mat"
    write_paderborn_mat(path, list(paderborn.CHANNELS))
    rec = paderborn.load_recording(path, decimate_by=4)
    assert rec.recording_id == "paderborn/KA01/N15_M07_F10/1"
    assert (rec.label, rec.origin, rec.fs) == ("outer", "artificial", 16_000.0)
    assert set(rec.signals) == set(paderborn.CHANNELS)
    lengths = {len(v) for v in rec.signals.values()}
    assert len(lengths) == 1  # channels truncated to a common length before decimation


def test_missing_channel_is_reported_with_what_was_found(tmp_path):
    path = tmp_path / "N15_M07_F10_KI01_2.mat"
    write_paderborn_mat(path, ["vibration_1", "phase_current_1"])
    with pytest.raises(ValueError, match="phase_current_2"):
        paderborn.load_mat(path)


def test_load_all_skips_bad_files_and_says_why(tmp_path):
    write_paderborn_mat(tmp_path / "N15_M07_F10_K001_1.mat", list(paderborn.CHANNELS))
    (tmp_path / "N15_M07_F10_K001_2.mat").write_bytes(b"not a mat file")
    recs, skipped = paderborn.load_all(tmp_path)
    assert len(recs) == 1
    assert len(skipped) == 1 and "K001_2" in skipped[0]


def test_cwru_prefers_the_files_own_channel(tmp_path):
    scipy.io.savemat(
        tmp_path / "105.mat",
        {
            "X105_DE_time": np.ones((100, 1)),
            "X999_DE_time": np.zeros((100, 1)),
            "X105_FE_time": np.zeros((100, 1)),
        },
    )
    rec = cwru.load_recording(tmp_path / "105.mat")
    assert rec.label == "inner" and rec.condition == "load_0hp"
    assert float(rec.signals["vibration_de"].mean()) == 1.0


def test_cwru_table():
    assert len(cwru.FILES) == 40
    assert cwru.FILES[97] == ("healthy", 0.0, 0)
    assert cwru.FILES[237] == ("outer", 0.021, 3)
