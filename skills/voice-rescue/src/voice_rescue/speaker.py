"""Is the restored voice still the same person? Speaker-embedding cosine similarity.

Model: WeSpeaker ECAPA-TDNN (512 channels, VoxCeleb2, Apache-2.0) as ONNX. Front end: Kaldi-style
80-bin log-mel fbank (25 ms / 10 ms, Hamming, pre-emphasis 0.97, no dither) on 16-bit-scaled
audio, mean-normalised over time, as in WeSpeaker's own inference code.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .install import component_path

SR = 16000


def _mel(f):
    return 1127.0 * np.log(1.0 + np.asarray(f) / 700.0)


@lru_cache(maxsize=1)
def _mel_banks(n_bins: int = 80, n_fft: int = 512, low: float = 20.0, high: float = SR / 2) -> np.ndarray:
    half = n_fft // 2
    width = SR / n_fft
    mlo, mhi = _mel(low), _mel(high)
    delta = (mhi - mlo) / (n_bins + 1)
    mf = _mel(width * np.arange(half))
    banks = np.zeros((n_bins, half + 1))
    for m in range(n_bins):
        left, center, right = mlo + m * delta, mlo + (m + 1) * delta, mlo + (m + 2) * delta
        up = (mf - left) / (center - left)
        down = (right - mf) / (right - center)
        banks[m, :half] = np.maximum(0.0, np.minimum(up, down))
    return banks


def fbank(x16: np.ndarray) -> np.ndarray:
    x = np.asarray(x16, dtype=np.float64) * 32768.0
    flen, fshift = 400, 160
    if len(x) < flen:
        x = np.pad(x, (0, flen - len(x)))
    n = 1 + (len(x) - flen) // fshift
    fr = np.lib.stride_tricks.sliding_window_view(x, flen)[::fshift][:n].copy()
    fr -= fr.mean(axis=1, keepdims=True)
    prev = np.concatenate([fr[:, :1], fr[:, :-1]], axis=1)
    fr = fr - 0.97 * prev
    fr *= np.hamming(flen)
    spec = np.abs(np.fft.rfft(fr, n=512, axis=1)) ** 2
    mel = spec @ _mel_banks().T
    return np.log(np.maximum(mel, np.finfo(np.float32).eps)).astype(np.float32)


@lru_cache(maxsize=1)
def _session():
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(str(component_path("ecapa")), sess_options=opts, providers=["CPUExecutionProvider"])


def embedding(x16: np.ndarray) -> np.ndarray:
    f = fbank(x16)
    f = f - f.mean(axis=0, keepdims=True)
    s = _session()
    name = s.get_inputs()[0].name
    e = s.run(None, {name: f[None, :, :]})[0][0]
    return e / (np.linalg.norm(e) + 1e-9)


def similarity(a16: np.ndarray, b16: np.ndarray) -> float:
    return float(np.dot(embedding(a16), embedding(b16)))
