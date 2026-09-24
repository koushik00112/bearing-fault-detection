from dataclasses import dataclass, field

import numpy as np


@dataclass
class Recording:
    """One continuous measurement. The unit that must never be split across train and test."""

    recording_id: str  # globally unique, e.g. "paderborn/KA01/N15_M07_F10/3"
    source: str  # "paderborn" | "cwru" | "synthetic"
    bearing: str  # physical bearing, e.g. "KA01"
    condition: str  # operating condition, e.g. "N15_M07_F10" or "load_2hp"
    label: str  # one of bearing.CLASSES (or source-specific for CWRU)
    origin: str  # "none" (healthy) | "artificial" | "real" | "synthetic"
    fs: float  # sampling rate in Hz, same for all channels
    signals: dict[str, np.ndarray] = field(repr=False)

    @property
    def n_samples(self) -> int:
        return min(len(s) for s in self.signals.values())
