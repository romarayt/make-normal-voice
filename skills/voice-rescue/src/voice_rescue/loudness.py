"""ITU-R BS.1770 loudness in numpy (mono), for windows where calling ffmpeg per window is too slow.

Values match ffmpeg's ebur128 integrated loudness for mono within ~0.1 LU (tested). The final
deliverable is always measured by ffmpeg on the real stereo file; this module is for analysis.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def _k_filter(sr: int):
    # BS.1770-4 pre-filter (high shelf) and RLB high-pass, designed for any rate via bilinear transform.
    f0, g, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = np.tan(np.pi * f0 / sr)
    vh = 10 ** (g / 20)
    vb = vh ** 0.4996667741545416
    a0 = 1 + k / q + k * k
    b1 = [(vh + vb * k / q + k * k) / a0, 2 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0]
    a1 = [1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]
    f0, q = 38.13547087602444, 0.5003270373238773
    k = np.tan(np.pi * f0 / sr)
    a0 = 1 + k / q + k * k
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]
    return (b1, a1), (b2, a2)


def block_power(x: np.ndarray, sr: int, block: float = 0.4, step: float = 0.1) -> np.ndarray:
    (b1, a1), (b2, a2) = _k_filter(sr)
    y = signal.lfilter(b2, a2, signal.lfilter(b1, a1, x.astype(np.float64)))
    n, h = int(block * sr), int(step * sr)
    if len(y) < n:
        return np.array([np.mean(y ** 2)]) if len(y) else np.zeros(0)
    c = np.concatenate([[0.0], np.cumsum(y ** 2)])
    starts = np.arange(0, len(y) - n + 1, h)
    return (c[starts + n] - c[starts]) / n


def integrated(x: np.ndarray, sr: int, channels: int = 1) -> float:
    """Gated integrated loudness. `channels=2` reports what a dual-mono copy of x would measure."""
    p = block_power(x, sr)
    if len(p) == 0:
        return float("-inf")
    p = p * channels
    lk = -0.691 + 10 * np.log10(p + 1e-20)
    p = p[lk > -70]
    if len(p) == 0:
        return float("-inf")
    rel = -0.691 + 10 * np.log10(p.mean()) - 10
    p = p[-0.691 + 10 * np.log10(p) > rel]
    return float(-0.691 + 10 * np.log10(p.mean())) if len(p) else float("-inf")


def windows(x: np.ndarray, sr: int, win: float = 10.0) -> np.ndarray:
    """Integrated (gated) loudness of consecutive windows: shape (n, 2) of [start_s, LUFS]."""
    n = int(win * sr)
    out = []
    for i in range(0, max(1, len(x) - n // 2), n):
        out.append((i / sr, integrated(x[i:i + n], sr)))
    return np.array(out)
