"""Thin wrappers around ffmpeg/ffprobe.

ffmpeg is always called with an argument list, never through a shell: file names with spaces,
quotes, `=` or Cyrillic letters are passed through unchanged.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REQUIRED_FILTERS = (
    "highpass", "deesser", "alimiter", "ebur128", "afade", "atrim", "pan", "aresample",
    "volume", "concat", "adeclip", "anlmdn", "bandreject",
)


class FFmpegError(RuntimeError):
    pass


def binary(name: str) -> str:
    env = os.environ.get(f"VOICE_RESCUE_{name.upper()}")
    path = env or shutil.which(name)
    if not path:
        raise FFmpegError(f"{name} not found on PATH; run `voice-rescue doctor` for install hints")
    return path


def run(args: list[str], *, input_bytes: bytes | None = None, check: bool = True) -> subprocess.CompletedProcess:
    cmd = [binary("ffmpeg"), "-hide_banner", "-nostdin", *args]
    if input_bytes is not None:
        cmd.remove("-nostdin")
    p = subprocess.run(cmd, input=input_bytes, capture_output=True)
    if check and p.returncode != 0:
        tail = p.stderr.decode("utf-8", "replace")[-2000:]
        raise FFmpegError(f"ffmpeg failed ({p.returncode}): {tail}")
    return p


def version() -> str:
    p = subprocess.run([binary("ffmpeg"), "-version"], capture_output=True, text=True)
    first = p.stdout.splitlines()[0] if p.stdout else ""
    m = re.search(r"version\s+(\S+)", first)
    return m.group(1) if m else first


def filters() -> set[str]:
    p = subprocess.run([binary("ffmpeg"), "-hide_banner", "-filters"], capture_output=True, text=True)
    names = set()
    for line in p.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and re.fullmatch(r"[A-Z.|]{2,4}", parts[0]):
            names.add(parts[1])
    return names


def probe(path: str | Path) -> dict:
    cmd = [binary("ffprobe"), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    p = subprocess.run(cmd, capture_output=True)
    out = p.stdout.decode("utf-8", "replace")
    data = json.loads(out) if out.strip() else {}
    data["_errors"] = [ln for ln in p.stderr.decode("utf-8", "replace").splitlines() if ln.strip()]
    data["_returncode"] = p.returncode
    return data


def audio_streams(info: dict) -> list[dict]:
    return [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]


def decode(path: str | Path, sr: int | None = None, channels: int | None = None, *, start: float | None = None,
           duration: float | None = None, stream: int = 0, af: str | None = None) -> tuple[np.ndarray, int]:
    """Decode one audio stream to float32, shape (frames, channels)."""
    if sr is None or channels is None:
        streams = audio_streams(probe(path))
        if not streams:
            raise FFmpegError(f"no audio stream in {path}")
        s = streams[min(stream, len(streams) - 1)]
        sr = sr or int(s["sample_rate"])
        channels = channels or int(s["channels"])
    args = ["-v", "error"]
    if start:
        args += ["-ss", f"{start:.6f}"]
    if duration:
        args += ["-t", f"{duration:.6f}"]
    args += ["-i", str(path), "-map", f"0:a:{stream}", "-vn"]
    if af:
        args += ["-af", af]
    args += ["-ac", str(channels), "-ar", str(sr), "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]
    p = run(args)
    x = np.frombuffer(p.stdout, dtype=np.float32).copy()
    return x.reshape(-1, channels), sr


def stream_blocks(path: str | Path, sr: int, channels: int, block_s: float = 30.0, stream: int = 0):
    """Yield float32 blocks of shape (frames, channels) without holding the whole file in memory."""
    cmd = [binary("ffmpeg"), "-hide_banner", "-nostdin", "-v", "error", "-i", str(path), "-map", f"0:a:{stream}",
           "-vn", "-ac", str(channels), "-ar", str(sr), "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    nbytes = int(block_s * sr) * channels * 4
    try:
        while True:
            buf = p.stdout.read(nbytes)
            if not buf:
                break
            usable = len(buf) - len(buf) % (4 * channels)
            yield np.frombuffer(buf[:usable], dtype=np.float32).reshape(-1, channels)
    finally:
        p.stdout.close()
        p.wait()


def decode_errors(path: str | Path, stream: int = 0, limit: int = 20) -> list[str]:
    """Fully decode the stream and collect decoder error lines (a broken file shows up here)."""
    p = run(["-v", "error", "-i", str(path), "-map", f"0:a:{stream}", "-f", "null", "-"], check=False)
    lines = [ln for ln in p.stderr.decode("utf-8", "replace").splitlines() if ln.strip()]
    if p.returncode != 0 and not lines:
        lines = [f"ffmpeg exited with {p.returncode}"]
    return lines[:limit]


@dataclass
class Loudness:
    integrated: float
    lra: float
    true_peak: float


def _parse_ebur128(stderr: str) -> Loudness:
    s = stderr[stderr.rfind("Summary:"):]
    def grab(label: str) -> float:
        m = re.search(label + r":\s+(-?inf|-?[\d.]+)", s)
        if not m:
            return float("nan")
        return float("-inf") if "inf" in m.group(1) else float(m.group(1))
    return Loudness(grab(r"\bI"), grab(r"\bLRA"), grab(r"\bPeak"))


def loudness(path: str | Path, *, af_prefix: str | None = None, start: float | None = None,
             end: float | None = None) -> Loudness:
    """EBU R128 integrated loudness, LRA and true peak of a file (as it is: stereo stays stereo)."""
    chain = []
    if af_prefix:
        chain.append(af_prefix)
    if start is not None or end is not None:
        trim = "atrim=" + ":".join(p for p in (f"start={start}" if start is not None else "",
                                                f"end={end}" if end is not None else "") if p)
        chain.insert(0, trim)
    chain.append("ebur128=peak=true:framelog=quiet")
    p = run(["-nostats", "-i", str(path), "-af", ",".join(chain), "-f", "null", "-"])
    return _parse_ebur128(p.stderr.decode("utf-8", "replace"))


def loudness_array(x: np.ndarray, sr: int) -> Loudness:
    """Same as loudness() for an in-memory signal of shape (frames,) or (frames, channels)."""
    x = np.asarray(x, dtype=np.float32)
    ch = 1 if x.ndim == 1 else x.shape[1]
    p = run(["-nostats", "-f", "f32le", "-ar", str(sr), "-ac", str(ch), "-i", "pipe:0",
             "-af", "ebur128=peak=true:framelog=quiet", "-f", "null", "-"], input_bytes=x.tobytes())
    return _parse_ebur128(p.stderr.decode("utf-8", "replace"))


def short_term_loudness(path: str | Path) -> np.ndarray:
    """Short-term (3 s) loudness every 100 ms, as an array of LUFS values."""
    p = run(["-nostats", "-i", str(path), "-af", "ebur128=framelog=verbose", "-f", "null", "-"])
    vals = []
    for m in re.finditer(r"t:\s*([\d.]+).*?S:\s*(-?[\d.]+|-inf)", p.stderr.decode("utf-8", "replace")):
        v = m.group(2)
        vals.append((float(m.group(1)), -120.0 if "inf" in v else float(v)))
    return np.array(vals, dtype=np.float64).reshape(-1, 2)
