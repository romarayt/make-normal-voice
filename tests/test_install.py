"""install/doctor: pinned and checksummed downloads, resume, ask once, no re-download, offline honesty."""

import hashlib
import io
import urllib.request

import pytest
from make_normal_voice import install, paths


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("MAKE_NORMAL_VOICE_HOME", str(tmp_path / "vr"))
    return tmp_path / "vr"


def test_registry_is_pinned_and_checksummed():
    for c in install.COMPONENTS.values():
        assert c.license and c.source and c.size > 0
        for f in c.files:
            assert "/resolve/main/" not in f.url and "/master/" not in f.url, f.url
            if f.dest.endswith((".pt", ".onnx", ".bin")):
                assert len(f.sha256) == 64, f.dest


class FakeResp(io.BytesIO):
    def __init__(self, data, status):
        super().__init__(data)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def fake_server(blob, seen):
    def urlopen(req, timeout=None):
        rng = req.headers.get("Range")
        seen.append(rng)
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
            return FakeResp(blob[start:], 206)
        return FakeResp(blob, 200)
    return urlopen


def test_download_resumes_and_verifies(tmp_path, monkeypatch):
    blob = bytes(range(256)) * 4096
    f = install.File("https://example.invalid/x.bin", hashlib.sha256(blob).hexdigest(), len(blob), "x/x.bin")
    dest = tmp_path / "x.bin"
    part = dest.with_suffix(".bin.part")
    part.write_bytes(blob[: len(blob) // 3])
    seen = []
    monkeypatch.setattr(urllib.request, "urlopen", fake_server(blob, seen))
    install._download(f, dest, log=lambda m: None)
    assert dest.read_bytes() == blob and not part.exists()
    assert seen == [f"bytes={len(blob) // 3}-"]


def test_bad_checksum_is_deleted(tmp_path, monkeypatch):
    blob = b"a" * 1000
    f = install.File("https://example.invalid/y.bin", "0" * 64, len(blob), "y.bin")
    monkeypatch.setattr(urllib.request, "urlopen", fake_server(blob, []))
    with pytest.raises(RuntimeError, match="sha256"):
        install._download(f, tmp_path / "y.bin", log=lambda m: None)
    assert not (tmp_path / "y.bin").exists() and not (tmp_path / "y.bin.part").exists()


def _fake_component(monkeypatch, blob):
    c = install.Component("fake", "fake model", "MIT", "https://example.invalid",
                          [install.File("https://example.invalid/f.bin", hashlib.sha256(blob).hexdigest(), len(blob),
                                        "fake/f.bin")], used_for="tests")
    monkeypatch.setitem(install.COMPONENTS, "fake", c)
    return c


def test_asks_once_and_respects_no(home, monkeypatch):
    blob = b"b" * 5000
    _fake_component(monkeypatch, blob)
    asked = []
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network")))
    ok = install.install(["fake"], ask=lambda q: asked.append(q) or "n", log=lambda m: None)
    assert not ok and len(asked) == 1
    assert not (paths.models_dir() / "fake" / "f.bin").exists()


def test_second_install_downloads_nothing(home, monkeypatch):
    blob = b"c" * 5000
    _fake_component(monkeypatch, blob)
    calls = []
    monkeypatch.setattr(urllib.request, "urlopen", fake_server(blob, calls))
    assert install.install(["fake"], yes=True, log=lambda m: None)
    n = len(calls)
    assert install.install(["fake"], yes=True, log=lambda m: None, ask=lambda q: pytest.fail("asked again"))
    assert len(calls) == n


def test_import_from_local_folder(home, monkeypatch, tmp_path):
    blob = b"d" * 7000
    _fake_component(monkeypatch, blob)
    src = tmp_path / "elsewhere"
    src.mkdir()
    (src / "whatever-name.bin").write_bytes(blob)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network")))
    assert install.install(["fake"], from_dir=src, log=lambda m: None)


def test_missing_model_says_how_to_get_it(home):
    with pytest.raises(install.MissingComponent, match="make-normal-voice install"):
        install.component_path("sidon-decoder-cpu")


def test_doctor_reports_without_crashing(home):
    rep = install.doctor()
    names = {i["name"] for i in rep["items"]}
    assert "ffmpeg" in names and "disk" in names and "model sidon-decoder-cpu" in names
