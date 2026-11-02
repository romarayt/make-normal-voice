"""Timbre matching to an already delivered track.

The long-term average spectrum of speech (frames louder than the median) is compared in
third-octave bands, normalised at 400-2500 Hz. The difference is smoothed, limited to +-6 dB and
applied as a linear-phase FIR whose delay is removed exactly, so the timing does not move.
(ffmpeg's firequalizer shifts the signal by its `delay` even with zero_phase in a WAV; this avoids
that trap entirely.) Two passes leave about +-0.4 dB.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

CENTERS = 1000 * 2 ** (np.arange(-14, 14) / 3)
REF_BAND = (400, 2500)


def ltas(x: np.ndarray, sr: int, lo: float = 50, hi: float = 17000, n: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Third-octave speech LTAS in dB, relative to the 400-2500 Hz mean."""
    hop = n
    w = np.hanning(n)
    acc = np.zeros(n // 2 + 1)
    cnt = 0
    block = hop * 256
    for i in range(0, max(1, len(x) - n), block):
        seg = x[i: i + block + n]
        if len(seg) < n:
            break
        fr = np.lib.stride_tricks.sliding_window_view(seg, n)[::hop] * w
        S = np.abs(np.fft.rfft(fr, axis=1)) ** 2
        e = S.sum(1)
        keep = e > np.percentile(e, 50)
        acc += S[keep].sum(0)
        cnt += int(keep.sum())
    f = np.fft.rfftfreq(n, 1 / sr)
    P = acc / max(cnt, 1)
    cs = CENTERS[(CENTERS > lo) & (CENTERS < min(hi, sr / 2 / 2 ** (1 / 6)))]
    b = np.array([P[(f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))].mean() for c in cs])
    b = 10 * np.log10(b + 1e-20)
    ref = b[(cs > REF_BAND[0]) & (cs < REF_BAND[1])].mean()
    return cs, b - ref


def correction(ref: np.ndarray, x: np.ndarray, sr: int, limit_db: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    cs, r = ltas(ref, sr)
    _, b = ltas(x, sr)
    d = r - b
    d = np.convolve(np.pad(d, 1, mode="edge"), [0.25, 0.5, 0.25], "valid")
    return cs, np.clip(d, -limit_db, limit_db)


def design_fir(cs: np.ndarray, gains_db: np.ndarray, sr: int, numtaps: int = 8191) -> np.ndarray:
    freqs = np.concatenate([[0.0], cs, [sr / 2]])
    g = np.concatenate([[gains_db[0]], gains_db, [gains_db[-1]]])
    # interpolate on a log-frequency grid so third-octave bands stay smooth
    grid = np.linspace(0, sr / 2, 4097)
    lg = np.log10(np.maximum(grid, 10.0))
    gi = np.interp(lg, np.log10(np.maximum(freqs, 10.0)), g)
    return signal.firwin2(numtaps, grid, 10 ** (gi / 20), fs=sr, window="hann")


def apply_fir(x: np.ndarray, h: np.ndarray, block: int = 1 << 20) -> np.ndarray:
    """Linear-phase FIR with its (numtaps-1)/2 delay removed: output is sample-aligned with input.

    Overlap-add in float32 blocks, so an hour at 48 kHz needs no float64 copy of the whole signal."""
    d = (len(h) - 1) // 2
    h = np.asarray(h, dtype=np.float32)
    n = len(x)
    y = np.zeros(n, dtype=np.float32)
    for i in range(0, n, block):
        seg = np.asarray(x[i: i + block], dtype=np.float32)
        conv = signal.fftconvolve(seg, h)            # covers output samples i - d ... i + len(seg) + len(h) - 1 - d
        lo, hi = i - d, i - d + len(conv)
        a, b = max(lo, 0), min(hi, n)
        if b > a:
            y[a:b] += conv[a - lo: b - lo]
    return y


def match(ref: np.ndarray, x: np.ndarray, sr: int, passes: int = 2) -> tuple[np.ndarray, list[dict]]:
    """Return x with the timbre of ref, plus the per-pass corrections for the report."""
    log = []
    y = x
    for _ in range(passes):
        cs, d = correction(ref, y, sr)
        h = design_fir(cs, d, sr)
        y = apply_fir(y, h)
        log.append({"bands_hz": [round(float(c)) for c in cs], "gain_db": [round(float(v), 2) for v in d]})
    return y, log


def residual(ref: np.ndarray, x: np.ndarray, sr: int, lo: float = 60, hi: float = 16000) -> tuple[np.ndarray, np.ndarray]:
    cs, r = ltas(ref, sr)
    _, b = ltas(x, sr)
    m = (cs >= lo) & (cs <= hi)
    return cs[m], (b - r)[m]
