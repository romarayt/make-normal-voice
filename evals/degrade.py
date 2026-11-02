"""Degrade clean references into the track classes of the router.

All degradations keep the timeline: room responses are aligned on their direct path, codecs are
re-aligned by cross-correlation, so a restored output can be scored against the clean reference.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from scipy import signal

SR = 48000


def load(path: Path, sr: int = SR) -> np.ndarray:
    x, s = sf.read(str(path), dtype="float32", always_2d=True)
    x = x[:, 0]
    return soxr.resample(x, s, sr, quality="VHQ").astype(np.float32) if s != sr else x


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-20))


def load_rir(path: Path) -> np.ndarray:
    h = load(path)
    k = int(np.argmax(np.abs(h)))
    h = h[k:]  # direct path at t = 0: the reverberant signal stays aligned with the clean one
    n = min(len(h), int(1.5 * SR))
    h = h[:n] * np.hanning(2 * n)[n:]
    return (h / (np.abs(h[0]) + 1e-9)).astype(np.float32)


def add_noise(x, noise, snr_db, rng):
    if len(noise) < len(x):
        noise = np.tile(noise, int(np.ceil(len(x) / len(noise))))
    i = int(rng.integers(0, len(noise) - len(x) + 1))
    n = noise[i: i + len(x)]
    n = n * (rms(x) / (rms(n) + 1e-12)) * 10 ** (-snr_db / 20)
    return (x + n).astype(np.float32)


def reverb(x, h):
    y = signal.fftconvolve(x, h)[: len(x)]
    return (y * rms(x) / (rms(y) + 1e-12)).astype(np.float32)


def ffmpeg_roundtrip(x, args_enc: list[str], tmp: Path) -> np.ndarray:
    a, b = tmp.with_suffix(".in.wav"), tmp.with_suffix(".codec")
    sf.write(str(a), x, SR, subtype="FLOAT")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(a), *args_enc, str(b)], check=True)
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(b), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    y = np.frombuffer(out, np.float32).copy()
    a.unlink()
    b.unlink()
    return realign(x, y)


def realign(ref, y, search=0.2):
    """Undo codec delay: shift y so it lines up with ref (sample-accurate cross-correlation)."""
    n = min(len(ref), len(y))
    m = int(search * SR)
    c = signal.correlate(y[: n], ref[: n], mode="full", method="fft")
    mid = n - 1
    k = int(np.argmax(c[mid - m: mid + m + 1])) - m
    y = y[k:] if k > 0 else np.concatenate([np.zeros(-k, np.float32), y])
    y = y[: len(ref)]
    return np.pad(y, (0, len(ref) - len(y))).astype(np.float32)


def music_bed(n, rng):
    t = np.arange(n) / SR
    out = np.zeros(n)
    chords = [[220.0, 277.2, 329.6], [196.0, 246.9, 293.7], [174.6, 220.0, 261.6], [164.8, 207.7, 246.9]]
    beat = 0.5
    for i in range(int(n / SR / (4 * beat)) + 1):
        ch = chords[i % 4]
        a, b = int(i * 4 * beat * SR), min(n, int((i + 1) * 4 * beat * SR))
        tt = t[a:b] - t[a]
        for f in ch:
            for h in (1, 2, 3):
                out[a:b] += np.sin(2 * np.pi * f * h * tt) / h * np.exp(-tt * 0.6)
        out[a:b] += 0.8 * np.sin(2 * np.pi * ch[0] / 2 * tt)
    hat = rng.standard_normal(n) * 0.0
    for k in range(int(n / SR / beat)):
        a = int(k * beat * SR)
        L = min(n - a, int(0.03 * SR))
        hat[a:a + L] += rng.standard_normal(L) * np.exp(-np.arange(L) / (0.008 * SR))
    hat = signal.sosfilt(signal.butter(2, 6000, "highpass", fs=SR, output="sos"), hat)
    return (out / rms(out) + 0.3 * hat / (rms(hat) + 1e-12)).astype(np.float32)


def lufs(x):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skills" / "voice-rescue" / "src"))
    from voice_rescue.loudness import integrated
    return integrated(x, SR)


CLASSES = ["clean", "far_echo", "far_echo_noise", "laptop", "noise_dry", "clipping", "phone", "codec_lowbit",
           "hum", "quiet", "dropouts", "music"]


def make(cls: str, x: np.ndarray, rng, rirs: dict, noises: list[np.ndarray], tmp: Path) -> tuple[np.ndarray, dict]:
    info = {"class": cls}
    if cls == "clean":
        y = x.copy()
    elif cls == "far_echo":
        name = rng.choice(rirs["far"])
        y = reverb(x, rirs["h"][name])
        y = add_noise(y, rng.standard_normal(len(y)).astype(np.float32), 35, rng)
        info["rir"] = name
    elif cls == "far_echo_noise":
        name = rng.choice(rirs["far"])
        y = reverb(x, rirs["h"][name])
        y = add_noise(y, noises[int(rng.integers(len(noises)))], 12, rng)
        info["rir"] = name
    elif cls == "laptop":
        name = rng.choice(rirs["near"])
        y = reverb(x, rirs["h"][name])
        y = signal.sosfilt(signal.butter(2, [150, 12000], "bandpass", fs=SR, output="sos"), y).astype(np.float32)
        pink = signal.lfilter([0.049922035, -0.095993537, 0.050612699, -0.004408786], [1, -2.494956002, 2.017265875, -0.522189400],
                              rng.standard_normal(len(y))).astype(np.float32)
        y = add_noise(y, pink, 22, rng)
        y = ffmpeg_roundtrip(y, ["-c:a", "aac", "-b:a", "96k", "-f", "adts"], tmp)
        info["rir"] = name
    elif cls == "noise_dry":
        y = add_noise(x, noises[int(rng.integers(len(noises)))], 5, rng)
    elif cls == "clipping":
        g = 0.5 / (np.percentile(np.abs(x), 99.0) + 1e-9)
        y = np.clip(x * g * 1.8, -0.5, 0.5).astype(np.float32)
        info["clip_ratio"] = float(np.mean(np.abs(y) >= 0.4999))
    elif cls == "phone":
        y = signal.sosfilt(signal.butter(4, [300, 3400], "bandpass", fs=SR, output="sos"), x).astype(np.float32)
        y = ffmpeg_roundtrip(y, ["-ar", "8000", "-c:a", "pcm_mulaw", "-f", "wav"], tmp)
    elif cls == "codec_lowbit":
        y = ffmpeg_roundtrip(x, ["-ar", "22050", "-ac", "1", "-c:a", "libmp3lame", "-b:a", "24k", "-f", "mp3"], tmp)
    elif cls == "hum":
        t = np.arange(len(x)) / SR
        f0 = 50.0 if rng.random() < 0.5 else 60.0
        h = sum(np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.28)) / k for k in (1, 2, 3, 4))
        y = (x + h / rms(h) * rms(x) * 10 ** (-20 / 20)).astype(np.float32)
        info["hum_hz"] = f0
    elif cls == "quiet":
        g = 10 ** ((-40 - lufs(x)) / 20)
        y = x * g + rng.standard_normal(len(x)).astype(np.float32) * 10 ** (-80 / 20)
        y = (np.round(y * 32767) / 32767).astype(np.float32)
    elif cls == "dropouts":
        y = x.copy()
        cuts = []
        for _ in range(6):
            a = int(rng.uniform(0.5, len(x) / SR - 0.7) * SR)
            n = int(rng.uniform(0.05, 0.15) * SR)
            y[a:a + n] = 0
            cuts.append([a / SR, n / SR])
        info["cuts"] = cuts
    elif cls == "music":
        m = music_bed(len(x), rng)
        y = (x + m * rms(x) * 10 ** (-12 / 20)).astype(np.float32)
    else:
        raise ValueError(cls)
    pk = np.max(np.abs(y))
    if pk > 0.99:
        y = y * (0.99 / pk)
    return y.astype(np.float32), info


def build(data: Path, out: Path, refs: list[dict], classes=CLASSES, seed: int = 7) -> list[dict]:
    rng = np.random.default_rng(seed)
    rm = json.loads((data / "rirs" / "meta.json").read_text())
    h = {n: load_rir(data / "rirs" / "rirs" / n) for n in rm["rirs"]}

    def t60(hh):  # crude: energy decay to -30 dB, doubled
        e = np.cumsum(hh[::-1] ** 2)[::-1]
        e = 10 * np.log10(e / e[0] + 1e-12)
        k = np.flatnonzero(e < -30)
        return 2 * k[0] / SR if len(k) else len(hh) / SR
    t = {n: t60(v) for n, v in h.items()}
    far = [n for n in h if t[n] >= 0.45]
    near = [n for n in h if 0.15 <= t[n] < 0.45]
    noises = [load(data / "rirs" / "noises" / n) for n in rm["noises"]]
    rir_set = {"h": h, "far": far or list(h), "near": near or list(h)}
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in refs:
        x = load(Path(r["path"]))
        x = x / (np.max(np.abs(x)) + 1e-9) * 0.5
        sf.write(str(out / f"{r['id']}__ref.wav"), x, SR, subtype="FLOAT")
        for cls in classes:
            y, info = make(cls, x, rng, rir_set, noises, out / f"tmp_{r['id']}")
            p = out / f"{r['id']}__{cls}.wav"
            sf.write(str(p), y, SR, subtype="FLOAT")
            rows.append({**r, **info, "input": str(p), "ref": str(out / f"{r['id']}__ref.wav")})
    (out / "set.json").write_text(json.dumps({"rows": rows, "t60": t}, indent=1, ensure_ascii=False))
    return rows
