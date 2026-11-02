"""Aligning recordings of the same speech by their loudness envelope.

Timecode, file names and device clocks are not trusted: in practice a recorder clock was 4 hours
off and camera clips had ~2 s gaps between files. What is trusted is the speech itself: the
log-energy envelope cross-correlation peak says how far apart two recordings are (coarse 10 ms
grid, refined on a 1 ms grid), and whether they are the same session at all (own session ~0.98,
someone else's recording 0.2-0.3).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import soxr

from .audio import log_envelope

SR = 16000


def _xcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Full cross-correlation c[k] = sum a[n+k] b[n] for k in [-(len(b)-1), len(a)-1]."""
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    c = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    return np.concatenate([c[-(len(b) - 1):], c[: len(a)]]) if len(b) > 1 else c[: len(a)]


def _refine(c: np.ndarray, k: int) -> float:
    if 0 < k < len(c) - 1:
        y0, y1, y2 = c[k - 1], c[k], c[k + 1]
        den = y0 - 2 * y1 + y2
        if den != 0:
            return k + 0.5 * (y0 - y2) / den
    return float(k)


@dataclass
class Offset:
    seconds: float   # where b[0] lies in a's timeline (b starts `seconds` after a starts)
    score: float     # normalised correlation peak, ~1 for the same speech


def coarse_offset(a: np.ndarray, b: np.ndarray, sr: int = SR, hop_s: float = 0.01) -> Offset:
    hop = int(round(hop_s * sr))
    ea, eb = log_envelope(a, hop), log_envelope(b, hop)
    if len(ea) < 2 or len(eb) < 2:
        return Offset(0.0, 0.0)
    c = _xcorr(ea, eb)
    k = int(np.argmax(c))
    lag = k - (len(eb) - 1)
    overlap = min(len(ea), lag + len(eb)) - max(0, lag)
    score = float(c[k] / max(overlap, 1))
    return Offset(lag * hop / sr, score)


def local_offset(a: np.ndarray, b: np.ndarray, sr: int, center_b: float, *, win: float = 20.0,
                 search: float = 0.05, hop_s: float = 0.001, guess: float = 0.0) -> Offset:
    """Offset of b against a around time `center_b` of b, assuming b[t] ~ a[t + guess].

    Returns where b's sample at `center_b - win/2` sits in a, as an absolute shift (seconds) so that
    a[t + shift] ~ b[t]. Uses a 1 ms envelope grid with parabolic refinement (~0.1 ms resolution).
    """
    hop = max(1, int(round(hop_s * sr)))
    b0 = int(round((center_b - win / 2) * sr))
    b0 = max(0, min(b0, len(b) - int(win * sr)))
    seg_b = b[b0: b0 + int(win * sr)]
    a0 = int(round((b0 / sr + guess - search) * sr))
    a1 = a0 + len(seg_b) + int(2 * search * sr)
    lo = max(0, a0)
    seg_a = a[lo: max(lo, a1)]
    ea, eb = log_envelope(seg_a, hop), log_envelope(seg_b, hop)
    if len(ea) <= len(eb) or len(eb) < 10:
        return Offset(float("nan"), 0.0)
    c = np.correlate(ea, eb, "valid")
    k = int(np.argmax(c))
    kk = _refine(c, k)
    shift = (lo + kk * hop) / sr - b0 / sr
    return Offset(shift, float(c[k] / len(eb)))


@dataclass
class Drift:
    offset: float          # seconds: a[t + offset + rate*t] ~ b[t]
    ppm: float             # clock-rate difference of b against a, parts per million
    residual_ms: float     # max residual of the linear fit
    points: list

    def to_dict(self):
        return asdict(self)


def gcc_phat(a: np.ndarray, b: np.ndarray, sr: int, center_b: float, *, win: float = 8.0, guess: float = 0.0,
             search: float = 0.1) -> Offset:
    """Shift s with a[t + s] ~ b[t] around time `center_b` of b, from the phase-transform cross-correlation.

    GCC-PHAT whitens the spectrum, so the peak sits on the *direct* sound even when a is a far,
    echoey recording of what b recorded close up. Envelope correlation is pulled late by the echo
    instead (+26 ms against a 0.6 s room in tests). Sub-sample precision by parabolic refinement.
    """
    nb = int(win * sr)
    b0 = int(round((center_b - win / 2) * sr))
    b0 = max(0, min(b0, len(b) - nb))
    seg_b = b[b0: b0 + nb].astype(np.float64)
    m = int(search * sr)
    a0 = int(round((b0 / sr + guess) * sr)) - m
    if a0 < 0 or a0 + nb + 2 * m > len(a) or len(seg_b) < nb:
        return Offset(float("nan"), 0.0)
    seg_a = a[a0: a0 + nb + 2 * m].astype(np.float64)
    n = 1 << int(np.ceil(np.log2(len(seg_a) + nb)))
    A, B = np.fft.rfft(seg_a * np.hanning(len(seg_a)), n), np.fft.rfft(seg_b * np.hanning(nb), n)
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-12
    r = np.fft.irfft(R, n)[: 2 * m + 1]  # lags 0 .. 2m of b inside seg_a
    k = int(np.argmax(r))
    kk = _refine(r, k)
    score = float(r[k] / (np.median(np.abs(r)) * 10 + 1e-12))
    return Offset((a0 + kk) / sr - b0 / sr, score)


def measure_drift(a: np.ndarray, b: np.ndarray, sr: int = SR, *, n_points: int = 12, win: float = 8.0,
                  guess: float | None = None, min_span: float = 300.0) -> Drift:
    """Offset and clock drift of b against a: envelope correlation for the coarse place, then
    GCC-PHAT at `n_points` places and a robust linear fit. The drift is fitted only when the points
    span at least `min_span` seconds: over a short clip a few ppm is far below a millisecond, and a
    fit there would turn measurement noise into a large fake drift."""
    if guess is None:
        guess = coarse_offset(a, b, sr).seconds
    dur_b = len(b) / sr
    usable_end = min(dur_b, (len(a) / sr) - guess)
    centers = np.linspace(win, max(win, usable_end - win), n_points)
    pts = []
    for cb in centers:
        o = gcc_phat(a, b, sr, cb, win=win, guess=guess, search=0.1)
        if np.isfinite(o.seconds) and o.score >= 1.0:
            pts.append((float(cb), float(o.seconds), o.score))
    if len(pts) < 3:
        # no clear direct-path peak (very different microphones or processing): 1 ms envelope grid
        pts = []
        for cb in centers:
            o = local_offset(a, b, sr, cb, win=20.0, guess=guess, search=0.08)
            if np.isfinite(o.seconds) and o.score >= 0.5:
                pts.append((float(cb), float(o.seconds), o.score))
    if not pts:
        return Drift(guess, 0.0, float("nan"), pts)
    t = np.array([p[0] for p in pts])
    s = np.array([p[1] for p in pts])
    # drop outliers (a cough heard by one source only) against the median offset
    keep = np.abs(s - np.median(s)) < 0.002 + 3 * np.median(np.abs(s - np.median(s)))
    t, s = t[keep], s[keep]
    if len(t) >= 3 and t.max() - t.min() >= min_span:
        for _ in range(2):
            k, c0 = np.polyfit(t, s, 1)
            r = s - (k * t + c0)
            if len(t) > 4 and np.max(np.abs(r)) > 0.001:
                i = int(np.argmax(np.abs(r)))
                t, s = np.delete(t, i), np.delete(s, i)
        k, c0 = np.polyfit(t, s, 1)
    else:
        k, c0 = 0.0, float(np.median(s))
    r = s - (k * t + c0)
    return Drift(float(c0), float(k * 1e6), float(np.max(np.abs(r)) * 1000) if len(r) else float("nan"), pts)


def stretch(x: np.ndarray, sr: int, ppm: float) -> np.ndarray:
    """Exact rate change by (1 + ppm*1e-6): what took T seconds now takes T*(1 + ppm*1e-6).

    With a drift fitted as a[t + offset + ppm*1e-6*t] ~ b[t], stretch(b, sr, ppm) shifted by `offset`
    lies on a's timeline. This is resampling (pitch moves by a few ppm, inaudible), not
    time-stretching, so nothing is smeared.
    """
    k = 1 + ppm * 1e-6
    return soxr.resample(x, sr, sr * k, quality="VHQ").astype(np.float32)


def place(x: np.ndarray, sr: int, offset: float, length: int) -> np.ndarray:
    """Put x on a timeline where its first sample sits at `offset` seconds; crop/pad to `length`."""
    out = np.zeros((length,) + x.shape[1:], dtype=np.float32)
    k = int(round(offset * sr))
    src0, dst0 = max(0, -k), max(0, k)
    n = min(len(x) - src0, length - dst0)
    if n > 0:
        out[dst0: dst0 + n] = x[src0: src0 + n]
    return out


def same_session(a: np.ndarray, b: np.ndarray, sr: int = SR, threshold: float = 0.6) -> tuple[bool, Offset]:
    o = coarse_offset(a, b, sr)
    return o.score >= threshold, o
