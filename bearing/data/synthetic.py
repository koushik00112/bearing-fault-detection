"""Synthetic bearing signals for tests, CI and the streaming demo. Never used for results.

A crude physical model: shaft harmonics plus noise. Outer- and inner-race faults add
decaying resonance bursts at the ball-pass frequencies (roughly 3.05x and 4.95x shaft
speed for a 6203 bearing); inner-race bursts are amplitude-modulated by shaft rotation.
Motor currents are a supply sinusoid with small fault-dependent sidebands. It's meant to
be separable, but not trivially.

The `source` field is always "synthetic". `origin` imitates Paderborn's artificial/real
labels only so that scenario D can be exercised in tests.
"""

import numpy as np

from bearing.data.recording import Recording

HEALTHY_TEST_BEARINGS = ("SH02", "SH03")
CONDITIONS = {"c1500": 25.0, "c900": 15.0, "c1500_low": 25.0, "c1500_f4": 25.0}
BPFO, BPFI = 3.05, 4.95


def _bursts(t: np.ndarray, rate: float, fs: float, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros_like(t)
    resonance = rng.uniform(2500, 3500)
    decay = rng.uniform(600, 900)
    period = 1.0 / rate
    for t0 in np.arange(rng.uniform(0, period), t[-1], period):
        jitter = t0 + rng.normal(0, 0.01 * period)  # slip
        idx = int(jitter * fs)
        n = min(len(t) - idx, int(0.01 * fs))
        if n <= 0:
            continue
        tau = np.arange(n) / fs
        out[idx : idx + n] += np.exp(-decay * tau) * np.sin(2 * np.pi * resonance * tau)
    return out


def make_recording(
    label: str,
    bearing: str,
    condition: str,
    rep: int,
    seed: int,
    fs: float = 16_000.0,
    seconds: float = 1.0,
    severity: float = 1.0,
    noise: float = 0.3,
    origin: str = "none",
) -> Recording:
    rng = np.random.default_rng(seed)
    shaft = CONDITIONS[condition]
    t = np.arange(int(fs * seconds)) / fs
    vib = (
        0.5 * np.sin(2 * np.pi * shaft * t + rng.uniform(0, 6.3))
        + 0.2 * np.sin(2 * np.pi * 2 * shaft * t)
        + noise * rng.standard_normal(len(t))
    )
    side = 0.0
    if label == "outer":
        vib += severity * _bursts(t, BPFO * shaft, fs, rng)
        side = BPFO * shaft
    elif label == "inner":
        mod = 1 + 0.6 * np.sin(2 * np.pi * shaft * t)
        vib += severity * mod * _bursts(t, BPFI * shaft, fs, rng)
        side = BPFI * shaft
    supply = 50.0
    currents = []
    for phase in (0.0, 2 * np.pi / 3):
        i = np.sin(2 * np.pi * supply * t + phase)
        if side:
            i += 0.02 * severity * np.sin(2 * np.pi * (supply + side) * t + phase)
        currents.append(i + 0.01 * rng.standard_normal(len(t)))
    return Recording(
        recording_id=f"synthetic/{bearing}/{condition}/{rep}",
        source="synthetic",
        bearing=bearing,
        condition=condition,
        label=label,
        origin=origin,
        fs=fs,
        signals={
            "vibration_1": vib.astype(np.float32),
            "phase_current_1": currents[0].astype(np.float32),
            "phase_current_2": currents[1].astype(np.float32),
        },
    )


def make_dataset(
    bearings_per_class: int = 3,
    reps: int = 4,
    conditions: tuple[str, ...] = tuple(CONDITIONS),
    seed: int = 0,
    seconds: float = 1.0,
) -> list[Recording]:
    """Small multi-bearing, multi-condition set shaped like Paderborn."""
    recs = []
    rng = np.random.default_rng(seed)
    for label, prefix in (("healthy", "SH"), ("inner", "SI"), ("outer", "SO")):
        for b in range(bearings_per_class):
            bearing = f"{prefix}{b:02d}"
            severity = float(rng.uniform(0.4, 1.2))  # each physical bearing differs
            # Mimic Paderborn's artificial/real split so scenario D can run on synthetic data.
            origin = "none" if label == "healthy" else ("artificial" if b % 2 == 0 else "real")
            for cond in conditions:
                for rep in range(reps):
                    s = int(rng.integers(1 << 31))
                    recs.append(
                        make_recording(
                            label,
                            bearing,
                            cond,
                            rep,
                            s,
                            seconds=seconds,
                            severity=severity,
                            origin=origin,
                        )
                    )
    return recs
