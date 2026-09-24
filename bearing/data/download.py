"""Download and unpack the raw datasets. Shows the total size and asks before downloading.

  python -m bearing.data.download paderborn --bearings K001,KA01 --dry-run
  python -m bearing.data.download paderborn --all            # ~4.6 GB (3-class bearings)
  python -m bearing.data.download cwru                       # ~40 files, a few MB each

Paderborn archives are .rar; they're unpacked with bsdtar (macOS built-in), unrar or 7z.
After unpacking, MANIFEST.sha256 records a hash of every .mat file so results can cite
exactly which data they came from.
"""

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import httpx

from bearing.data import cwru, paderborn


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def remote_size(client: httpx.Client, url: str) -> int | None:
    r = client.head(url, follow_redirects=True)
    if r.status_code != 200:
        return None
    return int(r.headers.get("content-length", 0)) or None


def fetch(client: httpx.Client, url: str, dest: Path, expected: int | None) -> None:
    if dest.exists() and (expected is None or dest.stat().st_size == expected):
        print(f"  have {dest.name}")
        return
    part = dest.with_suffix(dest.suffix + ".part")
    with client.stream("GET", url, follow_redirects=True) as r:
        r.raise_for_status()
        done = 0
        with part.open("wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if expected:
                    print(f"\r  {dest.name}: {done * 100 // expected:3d}%", end="", flush=True)
    if expected and part.stat().st_size != expected:
        raise RuntimeError(f"{dest.name}: got {part.stat().st_size} bytes, expected {expected}")
    part.rename(dest)
    print(f"\r  got  {dest.name} ({human(dest.stat().st_size)})")


def extract_rar(archive: Path, dest: Path) -> None:
    for cmd in (
        ["bsdtar", "-xf", str(archive), "-C", str(dest)],
        ["unrar", "x", "-o+", str(archive), str(dest) + "/"],
        ["7z", "x", "-y", f"-o{dest}", str(archive)],
    ):
        if shutil.which(cmd[0]):
            subprocess.run(cmd, check=True, capture_output=True)  # noqa: S603 - fixed tool list
            return
    sys.exit("No RAR extractor found. Install one of: bsdtar (libarchive), unrar, 7z.")


def write_manifest(root: Path) -> Path:
    lines = []
    for path in sorted(root.rglob("*.mat")):
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        lines.append(f"{h.hexdigest()}  {path.relative_to(root)}")
    manifest = root / "MANIFEST.sha256"
    manifest.write_text("\n".join(lines) + "\n")
    return manifest


def confirm(total: int, count: int, assume_yes: bool) -> bool:
    print(f"\n{count} file(s), {human(total)} to download.")
    if assume_yes:
        return True
    return input("Proceed? [y/N] ").strip().lower() == "y"


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m bearing.data.download")
    p.add_argument("dataset", choices=["paderborn", "cwru"])
    p.add_argument("--bearings", help="Paderborn codes, comma-separated")
    p.add_argument("--all", action="store_true", help="all 3-class Paderborn bearings")
    p.add_argument("--include-combined", action="store_true", help="also KB23/KB24/KB27")
    p.add_argument("--dest", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--yes", action="store_true", help="don't ask for confirmation")
    p.add_argument("--keep-archives", action="store_true")
    args = p.parse_args()

    if args.dataset == "paderborn":
        if args.all:
            codes = [
                c
                for c, (label, _) in paderborn.BEARINGS.items()
                if args.include_combined or label != "combined"
            ]
        elif args.bearings:
            codes = args.bearings.split(",")
            unknown = set(codes) - set(paderborn.BEARINGS)
            if unknown:
                p.error(f"unknown bearing codes: {sorted(unknown)}")
        else:
            p.error("choose --bearings or --all (the full set is several GB)")
        dest = args.dest or Path("data/raw/paderborn")
        files = [(f"{paderborn.BASE_URL}{c}.rar", dest / f"{c}.rar") for c in codes]
        print(
            "Paderborn KAt bearing data. Licence: CC BY-NC 4.0 (non-commercial, cite "
            "Lessmeier et al. 2016). See docs/data.md."
        )
    else:
        dest = args.dest or Path("data/raw/cwru")
        files = [(f"{cwru.BASE_URL}{n}.mat", dest / f"{n}.mat") for n in sorted(cwru.FILES)]
        print("CWRU bearing data (sanity check only). See docs/data.md.")

    dest.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=httpx.Timeout(30, read=300)) as client:
        sizes = [remote_size(client, url) for url, _ in files]
        for (url, _), size in zip(files, sizes, strict=True):
            print(f"  {url}  {human(size) if size else 'size unknown'}")
        if args.dry_run:
            print(f"\nDry run: {human(sum(s or 0 for s in sizes))} total, nothing downloaded.")
            return
        if not confirm(sum(s or 0 for s in sizes), len(files), args.yes):
            print("Cancelled.")
            return
        for (url, path), size in zip(files, sizes, strict=True):
            if path.suffix == ".rar" and (dest / path.stem).is_dir():
                print(f"  have {path.stem}/ (already unpacked)")
                continue
            fetch(client, url, path, size)

    if args.dataset == "paderborn":
        for _, archive in files:
            code = archive.stem
            if not (dest / code).is_dir() and archive.exists():
                print(f"  unpacking {archive.name}")
                extract_rar(archive, dest)
            if not args.keep_archives:
                archive.unlink(missing_ok=True)
    manifest = write_manifest(dest)
    print(f"Done. {sum(1 for _ in manifest.open())} files hashed in {manifest}")


if __name__ == "__main__":
    main()
