"""Case Western Reserve University bearing data: sanity check only.

Source: https://engineering.case.edu/bearingdatacenter (12k drive-end fault data and
normal baseline). CWRU is widely reported to be easy for ML models, so a high score here
shows only that the pipeline works, not that the model generalises.

One file per (fault, load), about 10 s each. Because there's only one recording per
cell, a recording-level split isn't possible; the sanity check uses leave-one-load-out.
"""

from pathlib import Path

import numpy as np
import scipy.io

from bearing.data.recording import Recording

BASE_URL = "https://engineering.case.edu/sites/default/files/"
FS = 12_000.0

# file number -> (label, fault size in inches, motor load in HP). 12k drive end, OR @6:00.
FILES: dict[int, tuple[str, float, int]] = {
    **{97 + hp: ("healthy", 0.0, hp) for hp in range(4)},
    **{105 + hp: ("inner", 0.007, hp) for hp in range(4)},
    **{118 + hp: ("ball", 0.007, hp) for hp in range(4)},
    **{130 + hp: ("outer", 0.007, hp) for hp in range(4)},
    **{169 + hp: ("inner", 0.014, hp) for hp in range(4)},
    **{185 + hp: ("ball", 0.014, hp) for hp in range(4)},
    **{197 + hp: ("outer", 0.014, hp) for hp in range(4)},
    **{209 + hp: ("inner", 0.021, hp) for hp in range(4)},
    **{222 + hp: ("ball", 0.021, hp) for hp in range(4)},
    **{234 + hp: ("outer", 0.021, hp) for hp in range(4)},
}

CLASSES = ("healthy", "inner", "ball", "outer")


def load_recording(path: Path) -> Recording:
    number = int(path.stem)
    label, size, hp = FILES[number]
    mat = scipy.io.loadmat(path)
    # Keys look like 'X105_DE_time'. A few files also carry another record's channels,
    # so prefer the key matching this file's number.
    de_keys = [k for k in mat if k.endswith("_DE_time")]
    if not de_keys:
        raise ValueError(f"{path.name}: no drive-end channel")
    own = [k for k in de_keys if k.startswith(f"X{number:03d}_")]
    key = own[0] if own else de_keys[0]
    signal = np.asarray(mat[key], dtype=np.float32).ravel()
    return Recording(
        recording_id=f"cwru/{number}",
        source="cwru",
        bearing=f"cwru-{label}-{size:.3f}",
        condition=f"load_{hp}hp",
        label=label,
        origin="artificial" if label != "healthy" else "none",
        fs=FS,
        signals={"vibration_de": signal},
    )


def load_all(root: Path) -> list[Recording]:
    return [
        load_recording(root / f"{n}.mat") for n in sorted(FILES) if (root / f"{n}.mat").exists()
    ]
