"""Paderborn University (KAt) bearing dataset.

Source: https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter
Licence: CC BY-NC 4.0. Non-commercial use; cite Lessmeier et al. (2016), see docs/data.md.
Do not commit raw data or anything that redistributes it.

Bearing tables follow Lessmeier et al., PHME 2016 (Tables 4-6). Each archive <CODE>.rar
unpacks to <CODE>/<CONDITION>_<CODE>_<REP>.mat, 20 repetitions of 4 s per condition.
"""

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io
from scipy.io.matlab import MatReadError
from scipy.signal import decimate

from bearing.data.recording import Recording

log = logging.getLogger(__name__)

BASE_URL = "https://groups.uni-paderborn.de/kat/BearingDataCenter/"
RAW_FS = 64_000.0

# code -> (label, origin). "combined" damage (KB*) has no artificial counterpart,
# so it's kept in the table but excluded from the 3-class experiments.
BEARINGS: dict[str, tuple[str, str]] = {
    **{f"K00{i}": ("healthy", "none") for i in range(1, 7)},
    # Artificial damage (EDM, engraving, drilling)
    **{
        c: ("outer", "artificial") for c in ["KA01", "KA03", "KA05", "KA06", "KA07", "KA08", "KA09"]
    },
    **{c: ("inner", "artificial") for c in ["KI01", "KI03", "KI05", "KI07", "KI08"]},
    # Real damage from accelerated lifetime tests
    **{c: ("outer", "real") for c in ["KA04", "KA15", "KA16", "KA22", "KA30"]},
    **{c: ("inner", "real") for c in ["KI04", "KI14", "KI16", "KI17", "KI18", "KI21"]},
    **{c: ("combined", "real") for c in ["KB23", "KB24", "KB27"]},
}

# Speed (rpm), load torque (Nm), radial force (N)
CONDITIONS: dict[str, tuple[int, float, int]] = {
    "N15_M07_F10": (1500, 0.7, 1000),
    "N09_M07_F10": (900, 0.7, 1000),
    "N15_M01_F10": (1500, 0.1, 1000),
    "N15_M07_F04": (1500, 0.7, 400),
}

CHANNELS = ("vibration_1", "phase_current_1", "phase_current_2")

# What a corrupt, truncated or unexpected file can raise; such files are skipped and listed.
# Real example: KA08/N15_M01_F10_KA08_2.mat raises TypeError ("Expecting matrix here")
# inside scipy's MAT reader (found 2026-09-25, the only unreadable file of 2,320).
READ_ERRORS = (
    ValueError,
    OSError,
    NotImplementedError,
    MatReadError,
    KeyError,
    AttributeError,
    TypeError,
)

FILENAME = re.compile(r"^(?P<cond>N\d\d_M\d\d_F\d\d)_(?P<code>K[AIB]?\d\d\d?)_(?P<rep>\d+)\.mat$")


def parse_filename(name: str) -> tuple[str, str, int]:
    """'N15_M07_F10_KA01_3.mat' -> ('N15_M07_F10', 'KA01', 3)."""
    m = FILENAME.match(name)
    if m is None:
        raise ValueError(f"not a Paderborn measurement file name: {name}")
    return m["cond"], m["code"], int(m["rep"])


def _field(obj: Any, name: str) -> Any:
    return getattr(obj, name) if hasattr(obj, name) else obj[name]


def load_mat(path: Path) -> dict[str, np.ndarray]:
    """Read the named channels from one measurement file.

    Layout (KAt documentation): a top-level struct named after the file, whose field `Y`
    is an array of structs with `Name` and `Data`. Channel names include vibration_1,
    phase_current_1, phase_current_2, speed, torque, force, temp_2_bearing_module.
    """
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    keys = [k for k in mat if not k.startswith("__")]
    if len(keys) != 1:
        raise ValueError(f"{path.name}: expected one top-level variable, found {keys}")
    entries = np.atleast_1d(_field(mat[keys[0]], "Y"))
    found: dict[str, np.ndarray] = {}
    for entry in entries:
        name = str(_field(entry, "Name")).strip()
        if name in CHANNELS:
            found[name] = np.asarray(_field(entry, "Data"), dtype=np.float32).ravel()
    missing = [c for c in CHANNELS if c not in found]
    if missing:
        names = [str(_field(e, "Name")) for e in entries]
        raise ValueError(f"{path.name}: missing channels {missing}; file has {names}")
    return found


def load_recording(path: Path, decimate_by: int = 4) -> Recording:
    cond, code, rep = parse_filename(path.name)
    if code not in BEARINGS:
        raise ValueError(f"unknown bearing code {code}")
    label, origin = BEARINGS[code]
    raw = load_mat(path)
    n = min(len(x) for x in raw.values())  # channels can differ by a few samples
    signals = {}
    for name, x in raw.items():
        x = x[:n]
        if decimate_by > 1:
            # Anti-aliased (zero-phase FIR) downsampling; drops content above fs/2.
            x = decimate(x, decimate_by, ftype="fir", zero_phase=True).astype(np.float32)
        signals[name] = x
    return Recording(
        recording_id=f"paderborn/{code}/{cond}/{rep}",
        source="paderborn",
        bearing=code,
        condition=cond,
        label=label,
        origin=origin,
        fs=RAW_FS / decimate_by,
        signals=signals,
    )


def iter_files(root: Path, bearings: list[str] | None = None) -> list[Path]:
    wanted = set(bearings) if bearings else set(BEARINGS)
    files = []
    for path in sorted(root.rglob("*.mat")):
        try:
            _, code, _ = parse_filename(path.name)
        except ValueError:
            continue
        if code in wanted:
            files.append(path)
    return files


def load_all(
    root: Path, bearings: list[str] | None = None, decimate_by: int = 4
) -> tuple[list[Recording], list[str]]:
    """Load every readable file. Returns (recordings, skipped file names with reasons)."""
    recordings, skipped = [], []
    for path in iter_files(root, bearings):
        try:
            recordings.append(load_recording(path, decimate_by))
        except READ_ERRORS as exc:
            log.warning("skipping %s: %s", path.name, exc)
            skipped.append(f"{path.name}: {exc}")
    return recordings, skipped
