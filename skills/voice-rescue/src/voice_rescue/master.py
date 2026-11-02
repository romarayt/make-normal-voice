"""Final mix: the chain that was proven on real recordings, with no compressor.

high-pass 70 Hz -> light de-esser -> one linear gain to the loudness target (measured on the final
dual-mono stereo, which reads 3 LU louder than mono) -> soft limiter with latency compensation ->
48 kHz / 24-bit dual-mono stereo.

A compressor is deliberately absent: it lifts the room tail between syllables back up (SRMR fell
below the untreated input in the first version). Stereo wideners on a single mic smear the phase.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import ffmpeg

PRE = "highpass=f=70:poles=2,deesser=i=0.2:m=0.5:f=0.55:s=o"
STEREO = "pan=stereo|c0=c0|c1=c0"


def limiter(limit: float) -> str:
    # latency=1 makes alimiter compensate its own look-ahead; without it the output is ~2 ms late.
    return f"alimiter=limit={limit:.4f}:attack=2:release=50:level=disabled:latency=1"


@dataclass
class MasterResult:
    path: str
    gain_db: float
    limit: float
    integrated: float
    true_peak: float
    lra: float
    passes: int

    def to_dict(self):
        return asdict(self)


def master(inp: str | Path, out: str | Path, *, target: float = -16.0, true_peak_max: float = -1.0,
           limit: float = 0.84, pre: str = PRE, max_passes: int = 4) -> MasterResult:
    """Mix a mono (or first-channel) track into the deliverable WAV."""
    inp, out = str(inp), str(out)
    first = ffmpeg.loudness(inp, af_prefix=f"aformat=channel_layouts=mono,{pre},{STEREO}")
    if not np.isfinite(first.integrated):
        raise ValueError(f"{inp}: no measurable loudness (silent or broken signal); refusing to apply a gain")
    gain = target - first.integrated
    m = first
    for n in range(1, max_passes + 1):  # noqa: B007 - n is reported
        chain = (f"aformat=channel_layouts=mono,{pre},volume={gain:.4f}dB:precision=double,"
                 f"{limiter(limit)},aresample=48000,{STEREO}")
        ffmpeg.run(["-v", "error", "-y", "-i", inp, "-af", chain, "-c:a", "pcm_s24le", out])
        m = ffmpeg.loudness(out)
        ok_i = abs(m.integrated - target) <= 0.1
        ok_tp = m.true_peak <= true_peak_max
        if ok_i and ok_tp:
            break
        if not ok_tp:
            limit *= 10 ** ((true_peak_max - 0.2 - m.true_peak) / 20)
        if not ok_i:
            gain += target - m.integrated
    return MasterResult(out, round(gain, 3), round(limit, 4), m.integrated, m.true_peak, m.lra, n)
