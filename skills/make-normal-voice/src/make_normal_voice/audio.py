"""Small signal helpers shared by the stages."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from . import ffmpeg


def load_mono(path: str | Path, sr: int = 16000, *, start: float | None = None,
              duration: float | None = None, af: str | None = None) -> np.ndarray:
    x, _ = ffmpeg.decode(path, sr=sr, channels=1, start=start, duration=duration, af=af)
    return x[:, 0]


def load_channel(path: str | Path, sr: int = 48000, channel: int = 0) -> np.ndarray:
    """One channel as is. (ffmpeg's -ac 1 downmix of a dual-mono file is 3 dB hotter than a channel.)"""
    x, _ = ffmpeg.decode(path, sr=sr, channels=1, af=f"pan=mono|c0=c{channel}")
    return x[:, 0]


def write_wav(path: str | Path, x: np.ndarray, sr: int, subtype: str = "PCM_24") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(x, dtype=np.float32), sr, subtype=subtype)
    return path


def dual_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[:, 0]
    return np.stack([x, x], axis=1)


def peak_normalize(x: np.ndarray, peak: float = 0.5) -> np.ndarray:
    return (x / (np.max(np.abs(x)) + 1e-9) * peak).astype(np.float32)


def log_envelope(x: np.ndarray, hop: int) -> np.ndarray:
    """Log-energy envelope, median-referenced and floored, z-scored. Robust for cross-correlation."""
    n = len(x) // hop
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    e = (x[: n * hop].astype(np.float64).reshape(n, hop) ** 2).mean(1)
    e = np.log10(e + 1e-10)
    e = np.clip(e - np.median(e), -1.0, None)
    return (e - e.mean()) / (e.std() + 1e-9)


def rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12))


def fade(y: np.ndarray, sr: int, fade_in: float = 0.0, fade_out: float = 0.0) -> np.ndarray:
    y = y.copy()
    if fade_in > 0:
        n = min(len(y), int(round(fade_in * sr)))
        y[:n] *= np.linspace(0.0, 1.0, n, dtype=np.float32).reshape(-1, *([1] * (y.ndim - 1)))
    if fade_out > 0:
        n = min(len(y), int(round(fade_out * sr)))
        y[len(y) - n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32).reshape(-1, *([1] * (y.ndim - 1)))
    return y
