"""WPE dereverberation (weighted prediction error; Nakatani et al. 2010), one channel, numpy only.

Per frequency bin, a linear filter predicts the late reverberation from the frames 2-31 hops in the
past and subtracts it. Nothing is generated: the output is the input minus a filtered copy of its own
past, so the voice keeps its own waveform, pitch and timbre. A restoration model re-synthesises the
voice instead, and on a real laptop recording a listener heard that as "a bit robotic" while every
gate passed (see references/pitfalls.md).

Settings were chosen by ear and measurement on that recording: 30 taps with a delay of 2 hops brought
the echo close to the restored track's, 10 taps left more room, and DNSMOS SIG did not drop with
either. On synthetic rooms SRMR goes 8.6 -> 10.6 at RT60 0.3 s, but only 5.6 -> 5.8 at 0.6 s with
DRR -3 dB: this is for a moderate room, not for a far camera mic. The signal is processed in 60 s chunks with 2 s linear crossfades; the solves are
batched matrix products (BLAS) in double precision, a few seconds per minute of audio and ~0.6 GB.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import istft, stft

SR = 48000
N_FFT, HOP = 1024, 256
TAPS, DELAY, ITERATIONS = 30, 2, 3
CHUNK_S, OVERLAP_S = 60.0, 2.0
BLOCK_BINS = 32          # frequency bins solved together: bounds the tap matrix to ~0.1 GB
REVISION = "f64-guard"   # part of the cache key: a result of an older engine is never reused
GUARD_RATIO = 2.0        # a bin may not come out with more energy than this times what went in
guarded = 0              # bins (per chunk) the guard returned to the input in the last dereverb()


def _wpe_bins(Y: np.ndarray, taps: int, delay: int, iterations: int) -> np.ndarray:
    """WPE on a block of frequency bins. Y: bins x frames.

    Always in double precision: the 1/power weights span ~12 orders of magnitude on a real recording
    (frame powers 126 dB apart in the 0-1.5 kHz block), and in complex64 the weighted covariance lost
    its small terms; the solve then returned garbage for the band that carries the voice (DNSMOS SIG
    3.60 -> 2.44 on one chunk, and different from run to run with the BLAS summation order)."""
    Y = np.asarray(Y, dtype=np.complex128)
    F, T = Y.shape
    Yt = np.zeros((F, T, taps), dtype=np.complex128)  # delayed past frames, one column per tap
    for k in range(taps):
        d = delay + k
        if d < T:
            Yt[:, d:, k] = Y[:, : T - d]
    eye = np.eye(taps, dtype=np.complex128)
    X = Y
    for _ in range(iterations):
        p = np.abs(X) ** 2
        lam = np.maximum(p, 1e-10 * float(p.max()) + 1e-20)                      # power of the dry estimate
        At = (Yt / lam[:, :, None]).transpose(0, 2, 1)                           # bins x taps x frames
        R = np.matmul(At, Yt.conj())                                             # weighted covariance
        P = np.matmul(At, Y.conj()[:, :, None])[..., 0]                          # weighted correlation
        R += (1e-6 * np.trace(R, axis1=1, axis2=2).real / taps)[:, None, None] * eye
        G = np.linalg.solve(R, P[..., None])[..., 0]
        X = Y - np.matmul(Yt, G.conj()[:, :, None])[..., 0]
    return guard(Y, X)


def guard(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """WPE only subtracts: a bin with more energy out than in means the solve diverged (seen in single
    precision as one 375 Hz bin 60x too loud). Such a bin is left as it came in."""
    global guarded
    bad = np.mean(np.abs(X) ** 2, axis=1) > GUARD_RATIO * np.mean(np.abs(Y) ** 2, axis=1) + 1e-30
    if bad.any():
        guarded += int(bad.sum())
        X = X.copy()
        X[bad] = Y[bad]
    return X


def dereverb_chunk(seg: np.ndarray, taps: int = TAPS, delay: int = DELAY, iterations: int = ITERATIONS) -> np.ndarray:
    """WPE on one stretch of 48 kHz mono audio; the output has exactly the input's length and timing."""
    _, _, Y = stft(np.asarray(seg, dtype=np.float64), fs=SR, window="hann", nperseg=N_FFT, noverlap=N_FFT - HOP)
    X = np.empty_like(Y)
    for f0 in range(0, Y.shape[0], BLOCK_BINS):
        X[f0:f0 + BLOCK_BINS] = _wpe_bins(Y[f0:f0 + BLOCK_BINS], taps, delay, iterations)
    _, z = istft(X, fs=SR, window="hann", nperseg=N_FFT, noverlap=N_FFT - HOP)
    z = z[: len(seg)]
    return np.pad(z, (0, len(seg) - len(z))).astype(np.float32)


def _starts(n: int, chunk: int) -> list[int]:
    """Chunk starts; a short remainder joins the chunk before it (a 3 s last chunk gave the filter too
    few frames to estimate: 31 % deviation from a one-piece run against 3-4 % for full chunks)."""
    st = list(range(0, n, chunk))
    if len(st) > 1 and n - st[-1] < chunk // 2:
        st.pop()
    return st


def spans(n: int, chunk_s: float = CHUNK_S, overlap_s: float = OVERLAP_S) -> list[tuple[int, int]]:
    chunk, ov = int(chunk_s * SR), int(overlap_s * SR)
    st = _starts(n, chunk)
    ends = st[1:] + [n]
    return [(max(0, a - ov), min(n, e + ov)) for a, e in zip(st, ends)]


def seams(duration: float, chunk_s: float = CHUNK_S) -> list[float]:
    """Centres of the chunk crossfades on the output timeline (s), for the artifacts gate."""
    return [a / SR for a in _starts(int(round(duration * SR)), int(chunk_s * SR))[1:]]


def dereverb(x: np.ndarray, *, taps: int = TAPS, delay: int = DELAY, iterations: int = ITERATIONS,
             chunk_s: float = CHUNK_S, overlap_s: float = OVERLAP_S, progress=None) -> np.ndarray:
    """WPE over a whole 48 kHz mono track in overlapping chunks; same length, sample-aligned."""
    global guarded
    guarded = 0
    x = np.asarray(x, dtype=np.float32)
    sp = spans(len(x), chunk_s, overlap_s)
    ov = int(overlap_s * SR)
    y = np.zeros(len(x), dtype=np.float64)
    wsum = np.zeros(len(x), dtype=np.float64)
    for i, (lo, hi) in enumerate(sp):
        z = dereverb_chunk(x[lo:hi], taps, delay, iterations)
        win = np.ones(hi - lo)
        n = min(2 * ov, hi - lo)
        if lo > 0:
            win[:n] = np.linspace(0.0, 1.0, n)
        if hi < len(x):
            win[-n:] = np.minimum(win[-n:], np.linspace(1.0, 0.0, n))
        y[lo:hi] += z * win
        wsum[lo:hi] += win
        if progress and ((i + 1) % 5 == 0 or i + 1 == len(sp)):
            progress(f"dereverb chunk {i + 1}/{len(sp)}")
    return (y / np.maximum(wsum, 1e-9)).astype(np.float32)
