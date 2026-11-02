"""`doctor` and `install`: what is present, what is missing, and fetching the missing parts.

Model weights are data. They are downloaded from pinned revisions, verified by sha256 and never
executed as code (TorchScript and ONNX files are loaded by torch/onnxruntime, not imported).
Downloads resume, show progress and time out instead of hanging.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from . import ffmpeg, paths

HF = "https://huggingface.co"
SIDON_REV = "b3b02d8bbd55fdbc410e6e46e76ef95ace4fbf52"
ECAPA_REV = "89426e111abc9ffcc40279baf808e087693c6e9b"
DNSMOS_REV = "82f1b17e7776a43eee395d0f45bae8abb700ad00"
WHISPER_SMALL_REV = "536b0662742c02347bc0e980a01041f333bce120"
WHISPER_TURBO_REV = "4df90f75321148c3a29a9e2351b7ddf8f5b115a8"


@dataclass
class File:
    url: str
    sha256: str
    size: int
    dest: str  # relative to models_dir()


@dataclass
class Component:
    id: str
    title: str
    license: str
    source: str
    files: list[File]
    group: str = "core"          # core | optional
    used_for: str = ""
    platforms: tuple[str, ...] = field(default=())

    @property
    def size(self) -> int:
        return sum(f.size for f in self.files)


def _hf(repo: str, rev: str, name: str) -> str:
    return f"{HF}/{repo}/resolve/{rev}/{name}"


COMPONENTS: dict[str, Component] = {c.id: c for c in [
    Component(
        "sidon-feature-extractor-cpu", "Sidon v0.1 feature extractor (w2v-BERT 2.0 fine-tune), CPU",
        "MIT", f"{HF}/sarulab-speech/sidon-v0.1",
        [File(_hf("sarulab-speech/sidon-v0.1", SIDON_REV, "feature_extractor_cpu.pt"),
              "fd9abc906a9048b3c047bd3d246a25f6485d09c2d6bb85e098f527f844efc019", 795498929,
              "sidon/feature_extractor_cpu.pt")],
        used_for="speech restoration (default engine)"),
    Component(
        "sidon-decoder-cpu", "Sidon v0.1 decoder (48 kHz vocoder), CPU", "MIT",
        f"{HF}/sarulab-speech/sidon-v0.1",
        [File(_hf("sarulab-speech/sidon-v0.1", SIDON_REV, "decoder_cpu.pt"),
              "34dcc80fab75bd1336369ba5b2063f6da475887aa824db94b495c4059c5fdea4", 209824877,
              "sidon/decoder_cpu.pt")],
        used_for="speech restoration (default engine)"),
    Component(
        "sidon-feature-extractor-cuda", "Sidon v0.1 feature extractor, CUDA", "MIT",
        f"{HF}/sarulab-speech/sidon-v0.1",
        [File(_hf("sarulab-speech/sidon-v0.1", SIDON_REV, "feature_extractor_cuda.pt"),
              "c3739c332da4ce00cc9019a328bcb19ee8f863ac5eb0b52386672772aab836ce", 795490336,
              "sidon/feature_extractor_cuda.pt")],
        group="optional", used_for="speech restoration on an NVIDIA GPU"),
    Component(
        "sidon-decoder-cuda", "Sidon v0.1 decoder, CUDA", "MIT", f"{HF}/sarulab-speech/sidon-v0.1",
        [File(_hf("sarulab-speech/sidon-v0.1", SIDON_REV, "decoder_cuda.pt"),
              "a789c71f54ca7db4b28becdc87c5e8b9260cb568cca462d502c54900db74a1cc", 209825094,
              "sidon/decoder_cuda.pt")],
        group="optional", used_for="speech restoration on an NVIDIA GPU"),
    Component(
        "dnsmos", "DNSMOS P.835 (sig_bak_ovr.onnx)", "CC-BY-4.0",
        "https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS",
        [File(f"https://raw.githubusercontent.com/microsoft/DNS-Challenge/{DNSMOS_REV}/DNSMOS/DNSMOS/sig_bak_ovr.onnx",
              "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd", 1157965,
              "dnsmos/sig_bak_ovr.onnx")],
        used_for="speech quality score without a reference (SIG/BAK/OVRL)"),
    Component(
        "ecapa", "WeSpeaker ECAPA-TDNN512 speaker embedding (VoxCeleb), ONNX", "Apache-2.0",
        f"{HF}/Wespeaker/wespeaker-voxceleb-ecapa-tdnn512",
        [File(_hf("Wespeaker/wespeaker-voxceleb-ecapa-tdnn512", ECAPA_REV, "voxceleb_ECAPA512.onnx"),
              "f901ae3571912bad4e11e157744cb8b7d77a0b46923b83c410bdbb6c77d69736", 24861931,
              "ecapa/voxceleb_ECAPA512.onnx")],
        used_for="checking that the voice is still the same person"),
    Component(
        "whisper-small", "faster-whisper small (CTranslate2)", "MIT", f"{HF}/Systran/faster-whisper-small",
        [File(_hf("Systran/faster-whisper-small", WHISPER_SMALL_REV, "model.bin"),
              "3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671", 483546902,
              "whisper-small/model.bin"),
         File(_hf("Systran/faster-whisper-small", WHISPER_SMALL_REV, "config.json"),
              "b55496ac7940a7ae47d2c01eab40edfd8701feec1229d9cce3b40014383fb828", 2370, "whisper-small/config.json"),
         File(_hf("Systran/faster-whisper-small", WHISPER_SMALL_REV, "tokenizer.json"),
              "fb7b63191e9bb045082c79fd742a3106a12c99513ab30df4a0d47fa6cb6fd0ab", 2203239,
              "whisper-small/tokenizer.json"),
         File(_hf("Systran/faster-whisper-small", WHISPER_SMALL_REV, "vocabulary.txt"),
              "34ce3fe1c5041027b3f8d42912270993f986dbc4bb34cf27f951e34a1e453913", 459861,
              "whisper-small/vocabulary.txt")],
        used_for="word check (WER) and finding the end of speech for trimming"),
    Component(
        "whisper-turbo", "faster-whisper large-v3-turbo (CTranslate2)", "MIT",
        f"{HF}/deepdml/faster-whisper-large-v3-turbo-ct2",
        [File(_hf("deepdml/faster-whisper-large-v3-turbo-ct2", WHISPER_TURBO_REV, "model.bin"),
              "e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da", 1617884929,
              "whisper-turbo/model.bin"),
         File(_hf("deepdml/faster-whisper-large-v3-turbo-ct2", WHISPER_TURBO_REV, "config.json"),
              "", 2263, "whisper-turbo/config.json"),
         File(_hf("deepdml/faster-whisper-large-v3-turbo-ct2", WHISPER_TURBO_REV, "preprocessor_config.json"),
              "", 340, "whisper-turbo/preprocessor_config.json"),
         File(_hf("deepdml/faster-whisper-large-v3-turbo-ct2", WHISPER_TURBO_REV, "tokenizer.json"),
              "", 2710337, "whisper-turbo/tokenizer.json"),
         File(_hf("deepdml/faster-whisper-large-v3-turbo-ct2", WHISPER_TURBO_REV, "vocabulary.json"),
              "", 1068114, "whisper-turbo/vocabulary.json")],
        group="optional", used_for="more accurate word check, mainly for non-English speech"),
]}

CORE = [c.id for c in COMPONENTS.values() if c.group == "core"]


class MissingComponent(RuntimeError):
    pass


def component_path(cid: str) -> Path:
    """Path of a single-file component, or the folder of a multi-file one. Raises if not installed."""
    c = COMPONENTS[cid]
    root = paths.models_dir()
    missing = [f for f in c.files if not _present(root / f.dest, f)]
    if missing:
        raise MissingComponent(
            f"{c.title} is not installed. Run `voice-rescue install {cid}` "
            f"({_mb(c.size)}); without network the step that needs it cannot run.")
    if len(c.files) == 1:
        return root / c.files[0].dest
    return (root / c.files[0].dest).parent


def _present(p: Path, f: File) -> bool:
    return p.is_file() and p.stat().st_size == f.size


def _mb(n: int) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def sha256(path: Path, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def _download(f: File, dest: Path, *, timeout: float = 30.0, retries: int = 5, log=print) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(f.url, headers={"User-Agent": "voice-rescue"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if have and r.status != 206:
                    have = 0  # server ignored Range: start over
                mode = "ab" if have else "wb"
                total = f.size
                t0, last, done = time.time(), 0.0, have
                with open(part, mode) as out:
                    while chunk := r.read(1 << 20):
                        out.write(chunk)
                        done += len(chunk)
                        now = time.time()
                        if now - last > 2 or done >= total:
                            speed = (done - have) / max(now - t0, 1e-6)
                            pct = 100 * done / total if total else 0
                            log(f"  {dest.name}: {pct:5.1f}% of {_mb(total)} at {speed / 1e6:.1f} MB/s")
                            last = now
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            if attempt == retries:
                raise RuntimeError(f"download failed for {f.url}: {e}") from e
            log(f"  {dest.name}: {e}; retry {attempt}/{retries - 1} (resuming)")
            time.sleep(min(30, 2 ** attempt))
    size = part.stat().st_size
    if size != f.size:
        raise RuntimeError(f"{dest.name}: got {size} bytes, expected {f.size}")
    if f.sha256:
        got = sha256(part)
        if got != f.sha256:
            part.unlink()
            raise RuntimeError(f"{dest.name}: sha256 mismatch ({got}); the partial file was removed")
    part.replace(dest)


def missing(ids: list[str]) -> list[Component]:
    root = paths.models_dir()
    out = []
    for cid in ids:
        c = COMPONENTS[cid]
        if any(not _present(root / f.dest, f) for f in c.files):
            out.append(c)
    return out


def install(ids: list[str] | None = None, *, yes: bool = False, from_dir: Path | None = None,
            log=print, ask=input) -> bool:
    """Install the given components (default: core). Returns True if everything is present."""
    ids = ids or CORE
    unknown = [i for i in ids if i not in COMPONENTS]
    if unknown:
        raise SystemExit(f"unknown component(s): {', '.join(unknown)}; known: {', '.join(COMPONENTS)}")
    todo = missing(ids)
    if not todo:
        log("Everything requested is already installed; nothing to download.")
        return True
    root = paths.models_dir()
    if from_dir:
        todo = _import_local(todo, Path(from_dir), root, log)
        if not todo:
            return True
    need = sum(c.size for c in todo)
    log("To download (into " + str(root) + "):")
    for c in todo:
        log(f"  - {c.title}: {_mb(c.size)}, license {c.license}, for {c.used_for}")
    log(f"Total {_mb(need)}.")
    free = paths.free_bytes(root)
    if free < need * 1.1 + 2e9:
        log(f"Not enough free disk space: {_mb(free)} free, need about {_mb(int(need * 1.1 + 2e9))}.")
        return False
    if not yes:
        ans = ask("Download now? [y/N] ").strip().lower()
        if ans not in ("y", "yes", "д", "да"):
            log("Nothing downloaded.")
            return False
    for c in todo:
        log(f"Downloading {c.title} from {c.source}")
        for f in c.files:
            d = root / f.dest
            if _present(d, f):
                continue
            _download(f, d, log=log)
    still = missing(ids)
    if still:
        log("Still missing: " + ", ".join(c.id for c in still))
        return False
    log("Done.")
    return True


def _import_local(todo: list[Component], src: Path, root: Path, log) -> list[Component]:
    """Take files with a matching size and sha256 from a local folder instead of downloading."""
    index: dict[int, list[Path]] = {}
    for p in src.rglob("*"):
        if p.is_file():
            index.setdefault(p.stat().st_size, []).append(p)
    left = []
    for c in todo:
        ok = True
        for f in c.files:
            d = root / f.dest
            if _present(d, f):
                continue
            hit = None
            for cand in index.get(f.size, []):
                if not f.sha256 or sha256(cand) == f.sha256:
                    hit = cand
                    break
            if hit is None:
                ok = False
                continue
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(hit, d)
            log(f"  copied {hit.name} from {src} (sha256 verified)")
        if not ok:
            left.append(c)
    return left


# ---------------------------------------------------------------- doctor

PACKAGES = ["numpy", "scipy", "soundfile", "soxr", "torch", "transformers", "gammatone", "onnxruntime",
            "faster_whisper", "matplotlib", "platformdirs"]


def ffmpeg_hint() -> str:
    sysname = platform.system()
    if sysname == "Darwin":
        return "brew install ffmpeg"
    if sysname == "Linux":
        return "sudo apt-get install ffmpeg   (or your distribution's package manager)"
    if sysname == "Windows":
        return "winget install Gyan.FFmpeg"
    return "install ffmpeg 6 or newer from https://ffmpeg.org/download.html"


def doctor(verify_hashes: bool = False) -> dict:
    rep: dict = {"ok": True, "items": []}

    def item(name, ok, detail, fix=""):
        rep["items"].append({"name": name, "ok": bool(ok), "detail": detail, "fix": fix})
        if not ok:
            rep["ok"] = False

    item("python", sys.version_info[:2] == (3, 12), f"{platform.python_version()} at {sys.executable}",
         "run through the bundled launcher, which uses a uv-managed Python 3.12")
    item("uv", shutil.which("uv") is not None, shutil.which("uv") or "not found",
         "curl -LsSf https://astral.sh/uv/install.sh | sh   (or: brew install uv)")
    try:
        v = ffmpeg.version()
        have = ffmpeg.filters()
        lack = [f for f in ffmpeg.REQUIRED_FILTERS if f not in have]
        ok = not lack and shutil.which("ffprobe") is not None
        item("ffmpeg", ok, f"{v}; " + ("all filters present" if not lack else "missing filters: " + ", ".join(lack)),
             ffmpeg_hint())
    except ffmpeg.FFmpegError as e:
        item("ffmpeg", False, str(e), ffmpeg_hint())
    for mod in PACKAGES:
        try:
            m = importlib.import_module(mod)
            item(f"package {mod}", True, getattr(m, "__version__", "present"))
        except Exception as e:  # noqa: BLE001 - report any import problem
            item(f"package {mod}", False, f"{type(e).__name__}: {e}",
                 "the launcher installs locked packages on first run; re-run it with network")
    root = paths.models_dir()
    for c in COMPONENTS.values():
        present = all(_present(root / f.dest, f) for f in c.files)
        detail = "installed" if present else f"missing ({_mb(c.size)})"
        if present and verify_hashes:
            bad = [f.dest for f in c.files if f.sha256 and sha256(root / f.dest) != f.sha256]
            if bad:
                present, detail = False, "checksum mismatch: " + ", ".join(bad)
        if c.group == "core":
            item(f"model {c.id}", present, detail, f"voice-rescue install {c.id}")
        else:
            rep["items"].append({"name": f"model {c.id} (optional)", "ok": present, "detail": detail,
                                 "fix": f"voice-rescue install {c.id}", "optional": True})
    free = paths.free_bytes(paths.home())
    item("disk", free > 5e9, f"{_mb(free)} free at {paths.home()}", "free some disk space (5 GB+ recommended)")
    rep["data_dir"] = str(paths.home())
    return rep


def try_install_ffmpeg(yes: bool, log=print, ask=input) -> bool:
    """Install ffmpeg with a package manager that needs no sudo (Homebrew, winget). Otherwise print how."""
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return True
    sysname = platform.system()
    cmd = None
    if sysname == "Darwin" and shutil.which("brew"):
        cmd = ["brew", "install", "ffmpeg"]
    elif sysname == "Windows" and shutil.which("winget"):
        cmd = ["winget", "install", "-e", "--id", "Gyan.FFmpeg"]
    if cmd is None:
        log("ffmpeg is missing. Install it with: " + ffmpeg_hint())
        return False
    if not yes and ask(f"ffmpeg is missing. Run `{' '.join(cmd)}`? [y/N] ").strip().lower() not in ("y", "yes", "д", "да"):
        return False
    return subprocess.run(cmd, env={**os.environ}).returncode == 0
