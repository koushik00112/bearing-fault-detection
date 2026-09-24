"""Turn raw recordings into a cached window dataset on disk.

  python -m bearing.prepare paderborn --raw data/raw/paderborn --out data/processed/paderborn
  python -m bearing.prepare cwru      --raw data/raw/cwru      --out data/processed/cwru
  python -m bearing.prepare synthetic                          --out data/processed/synthetic

Recordings are streamed one at a time, so the full Paderborn set never has to fit in RAM.
Output: windows.f32 (raw float32, memory-mapped when loaded), meta.csv, dataset.json.
"""

import argparse
import hashlib
import json
import logging
import subprocess
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from bearing.data import cwru, paderborn, synthetic
from bearing.data.recording import Recording
from bearing.windowing import META_COLUMNS, WindowSet, windows_of

log = logging.getLogger("prepare")


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607 - git from PATH is intended
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _manifest_hash(raw: Path | None) -> str | None:
    """Hash of the download manifest, tying results to the exact raw files used."""
    if raw is None or not (raw / "MANIFEST.sha256").exists():
        return None
    return hashlib.sha256((raw / "MANIFEST.sha256").read_bytes()).hexdigest()


def write_dataset(
    recordings: Iterator[Recording],
    out: Path,
    channels: tuple[str, ...],
    window: int,
    stride: int,
    max_windows: int | None,
    info: dict[str, object],
) -> WindowSet:
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    fs: float | None = None
    n = 0
    with (out / "windows.f32").open("wb") as f:
        for rec in recordings:
            if fs is None:
                fs = rec.fs
            elif rec.fs != fs:
                raise ValueError(f"mixed sampling rates ({fs}, {rec.fs})")
            X, r = windows_of(rec, channels, window, stride, max_windows)
            f.write(np.ascontiguousarray(X, dtype=np.float32).tobytes())
            rows.extend(r)
            n += len(X)
    if fs is None:
        raise SystemExit("no recordings found; check --raw and that archives are extracted")
    pd.DataFrame(rows, columns=META_COLUMNS).to_csv(out / "meta.csv", index=False)
    (out / "dataset.json").write_text(
        json.dumps(
            {
                "fs": fs,
                "channels": list(channels),
                "window": window,
                "stride": stride,
                "max_windows_per_recording": max_windows,
                "n_windows": n,
                "dtype": "float32",
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "code_version": _git_sha(),
                **info,
            },
            indent=2,
        )
    )
    return load(out)


def load(directory: Path) -> WindowSet:
    info = json.loads((directory / "dataset.json").read_text())
    shape = (info["n_windows"], len(info["channels"]), info["window"])
    X = np.memmap(directory / "windows.f32", dtype=np.float32, mode="r", shape=shape)
    meta = pd.read_csv(directory / "meta.csv", dtype={"bearing": str, "condition": str})
    if len(meta) != shape[0]:
        raise ValueError(f"{directory}: meta has {len(meta)} rows, windows file has {shape[0]}")
    return WindowSet(X=X, meta=meta, channels=tuple(info["channels"]), fs=info["fs"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="python -m bearing.prepare")
    p.add_argument("source", choices=["paderborn", "cwru", "synthetic"])
    p.add_argument("--raw", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--decimate", type=int, default=4, help="Paderborn: 64 kHz / 4 = 16 kHz")
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--stride", type=int, default=1024, help="window/2 = 50%% overlap")
    p.add_argument("--max-windows", type=int, default=16, help="per recording")
    p.add_argument("--bearings", help="comma-separated Paderborn codes (default: all 3-class)")
    args = p.parse_args()

    info: dict[str, object] = {
        "source": args.source,
        "raw_manifest_sha256": _manifest_hash(args.raw),
    }
    if args.source == "paderborn":
        if args.raw is None:
            p.error("--raw is required for paderborn")
        codes = (
            args.bearings.split(",")
            if args.bearings
            else [c for c, (label, _) in paderborn.BEARINGS.items() if label != "combined"]
        )
        skipped: list[str] = []

        def stream() -> Iterator[Recording]:
            for path in paderborn.iter_files(args.raw, codes):
                try:
                    yield paderborn.load_recording(path, args.decimate)
                except paderborn.READ_ERRORS as exc:
                    log.warning("skipping %s: %s", path.name, exc)
                    skipped.append(f"{path.name}: {exc}")

        info.update(
            {
                "decimate": args.decimate,
                "bearings": codes,
                "skipped_files": skipped,
                "licence": "CC BY-NC 4.0, cite Lessmeier et al. 2016",
            }
        )
        ws = write_dataset(
            stream(), args.out, paderborn.CHANNELS, args.window, args.stride, args.max_windows, info
        )
        # `skipped` is filled while streaming; rewrite the info with the final list.
        meta = json.loads((args.out / "dataset.json").read_text())
        meta["skipped_files"] = skipped
        (args.out / "dataset.json").write_text(json.dumps(meta, indent=2))
    elif args.source == "cwru":
        if args.raw is None:
            p.error("--raw is required for cwru")
        ws = write_dataset(
            iter(cwru.load_all(args.raw)),
            args.out,
            ("vibration_de",),
            args.window,
            args.stride,
            args.max_windows,
            info,
        )
    else:
        info["note"] = "SYNTHETIC: for tests and demos only, never for reported results"
        info["healthy_test_bearings"] = list(synthetic.HEALTHY_TEST_BEARINGS)
        recs = synthetic.make_dataset(bearings_per_class=4, reps=5, seconds=1.0)
        ws = write_dataset(
            iter(recs),
            args.out,
            paderborn.CHANNELS,
            args.window,
            args.stride,
            args.max_windows,
            info,
        )
    counts = ws.meta.groupby(["label", "origin"]).size()
    log.info(
        "wrote %d windows from %d recordings to %s\n%s",
        len(ws),
        ws.meta["recording_id"].nunique(),
        args.out,
        counts.to_string(),
    )


if __name__ == "__main__":
    main()
