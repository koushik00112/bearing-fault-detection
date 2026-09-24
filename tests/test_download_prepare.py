import hashlib
import json
import sys

import httpx
import pytest

from bearing import prepare
from bearing.data import download


def test_human_sizes():
    assert download.human(512) == "512.0 B"
    assert download.human(160 * 1024**2) == "160.0 MB"


def test_fetch_streams_to_part_then_renames_and_checks_size(tmp_path):
    payload = b"x" * 5000
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=payload))
    with httpx.Client(transport=transport) as c:
        download.fetch(c, "http://h/K001.rar", tmp_path / "K001.rar", expected=5000)
        assert (tmp_path / "K001.rar").read_bytes() == payload
        assert not (tmp_path / "K001.rar.part").exists()
        with pytest.raises(RuntimeError, match="expected 9999"):
            download.fetch(c, "http://h/K002.rar", tmp_path / "K002.rar", expected=9999)


def test_existing_complete_file_is_not_fetched_again(tmp_path):
    (tmp_path / "a.mat").write_bytes(b"12345")

    def fail(_req):
        raise AssertionError("should not download")

    with httpx.Client(transport=httpx.MockTransport(fail)) as c:
        download.fetch(c, "http://h/a.mat", tmp_path / "a.mat", expected=5)


def test_manifest_hashes_every_mat_file(tmp_path):
    (tmp_path / "K001").mkdir()
    (tmp_path / "K001" / "N15_M07_F10_K001_1.mat").write_bytes(b"abc")
    manifest = download.write_manifest(tmp_path).read_text().splitlines()
    assert manifest == [f"{hashlib.sha256(b'abc').hexdigest()}  K001/N15_M07_F10_K001_1.mat"]


def test_confirm_requires_an_explicit_yes(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    assert download.confirm(10, 1, assume_yes=False) is False
    monkeypatch.setattr("builtins.input", lambda _prompt: "y")
    assert download.confirm(10, 1, assume_yes=False) is True
    assert download.confirm(10, 1, assume_yes=True) is True


def test_paderborn_download_needs_an_explicit_selection(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["download", "paderborn"])
    with pytest.raises(SystemExit):
        download.main()


def test_prepare_cli_writes_a_labelled_synthetic_dataset(tmp_path, monkeypatch):
    out = tmp_path / "syn"
    monkeypatch.setattr(
        sys, "argv", ["prepare", "synthetic", "--out", str(out), "--max-windows", "2"]
    )
    prepare.main()
    info = json.loads((out / "dataset.json").read_text())
    assert info["source"] == "synthetic" and "SYNTHETIC" in info["note"]
    ws = prepare.load(out)
    assert len(ws) == info["n_windows"] > 0
    assert ws.meta.groupby("recording_id").size().max() <= 2


def test_load_detects_a_truncated_windows_file(tmp_path, recordings):
    prepare.write_dataset(
        iter(recordings[:2]), tmp_path, ("vibration_1",), 2048, 1024, None, {"source": "synthetic"}
    )
    info = json.loads((tmp_path / "dataset.json").read_text())
    info["n_windows"] += 1
    (tmp_path / "dataset.json").write_text(json.dumps(info))
    with pytest.raises(ValueError):
        prepare.load(tmp_path)
