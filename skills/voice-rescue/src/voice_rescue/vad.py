"""Speech detection with Silero VAD (MIT), the ONNX model bundled in the faster-whisper wheel."""

from __future__ import annotations

import numpy as np

SR = 16000


def speech_segments(x16: np.ndarray, *, threshold: float = 0.5, min_silence_ms: int = 300,
                    pad_ms: int = 30) -> list[tuple[float, float]]:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    opts = VadOptions(threshold=threshold, min_silence_duration_ms=min_silence_ms, speech_pad_ms=pad_ms,
                      min_speech_duration_ms=150)
    ts = get_speech_timestamps(np.asarray(x16, np.float32), vad_options=opts, sampling_rate=SR)
    return [(t["start"] / SR, t["end"] / SR) for t in ts]


def speech_mask(segments: list[tuple[float, float]], n: int, sr: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for a, b in segments:
        m[int(a * sr): int(b * sr)] = True
    return m


def speech_ratio(segments: list[tuple[float, float]], duration: float) -> float:
    return float(sum(b - a for a, b in segments) / duration) if duration > 0 else 0.0


def pauses(segments: list[tuple[float, float]], duration: float, min_len: float = 0.25) -> list[tuple[float, float]]:
    out, t = [], 0.0
    for a, b in segments:
        if a - t >= min_len:
            out.append((t, a))
        t = b
    if duration - t >= min_len:
        out.append((t, duration))
    return out


def nearest_pause(segments: list[tuple[float, float]], duration: float, t: float, radius: float = 10.0) -> float:
    """Middle of the longest pause within +-radius of t (t itself if there is none)."""
    best, best_len = t, 0.0
    for a, b in pauses(segments, duration, min_len=0.15):
        if b < t - radius or a > t + radius:
            continue
        lo, hi = max(a, t - radius), min(b, t + radius)
        if hi - lo > best_len:
            best, best_len = (lo + hi) / 2, hi - lo
    return best


def speech_windows(segments: list[tuple[float, float]], duration: float, win: float = 60.0,
                   n: int = 3) -> list[tuple[float, float]]:
    """Up to n windows (start, middle, end of the file) with the most speech in each region."""
    if duration <= win:
        return [(0.0, duration)]
    m = speech_mask(segments, int(duration * 10) + 1, 10)  # 100 ms grid
    w = int(win * 10)
    c = np.concatenate([[0], np.cumsum(m)])
    starts = np.arange(0, len(m) - w + 1)
    dens = c[starts + w] - c[starts]
    regions = np.array_split(starts, n)
    out = []
    for r in regions:
        if len(r) == 0:
            continue
        k = r[int(np.argmax(dens[r]))]
        out.append((k / 10, k / 10 + win))
    # keep distinct windows only
    uniq = []
    for a, b in out:
        if all(abs(a - u[0]) >= win for u in uniq):
            uniq.append((a, b))
    return uniq
