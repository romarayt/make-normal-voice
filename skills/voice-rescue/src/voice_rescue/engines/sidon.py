"""Sidon speech restoration (sarulab-speech/sidon-v0.1, MIT) with sample-exact chunking.

Sidon rebuilds the voice from w2v-BERT 2.0 features as if it had been recorded by a close
microphone, at 48 kHz. The reference demo processes 96 s chunks and glues the decoder outputs
together. Two things in that recipe shift the timing, and both are fixed here:

* The decoder returns ``frames * 960 - 26`` samples: the last 26 samples of every call are
  missing (the output is aligned at the start). Gluing 96 s chunks therefore loses 26 samples per
  chunk, which is 5.64 ppm, about 20 ms per hour. Here every chunk is decoded with context frames
  on both sides and cropped to exactly ``frames * 960`` samples.
* The decoder output for feature frame ``j`` sits ``LATENCY_48K`` samples later than the speech
  that produced it. The constant was measured on real recordings (see DECISIONS.md) and is removed
  so that the output lines up with the input.

Chunks are glued with a short crossfade inside the shared context, so a chunk seam has no click.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf

from .. import ffmpeg, memory
from ..install import component_path

SR_IN = 16000
SR_OUT = 48000
HOP_IN = 320          # 20 ms feature frame at 16 kHz
HOP_OUT = 960         # 20 ms feature frame at 48 kHz
DECODER_TAIL_LOSS = 26
LATENCY_48K = 354     # 7.375 ms, measured on a laptop mic and a lavalier (median of 25 windows)
CHUNK_FRAMES = 1500   # 30 s: same quality as the demo's 96 s, a quarter of the memory (see memory.py)
CONTEXT_FRAMES = 150  # 3 s of real audio on each side of a chunk
XFADE_OUT = 480       # 10 ms crossfade at chunk seams (48 kHz samples)
DEC_FRAMES = 150      # the decoder runs on 3 s of features at a time ...
DEC_CONTEXT = 16      # ... with 0.32 s of features on each side (its reach is about one frame)


def _load(device: str):
    import torch
    import transformers

    suffix = "cuda" if device.startswith("cuda") else "cpu"
    fe = torch.jit.load(str(component_path(f"sidon-feature-extractor-{suffix}")), map_location=device).eval()
    dec = torch.jit.load(str(component_path(f"sidon-decoder-{suffix}")), map_location=device).eval()
    # facebook/w2v-bert-2.0 preprocessor_config.json, written out so no download is needed.
    pre = transformers.SeamlessM4TFeatureExtractor(
        feature_size=80, num_mel_bins=80, padding_value=1.0, sampling_rate=SR_IN, stride=2)
    return fe, dec, pre


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def prepare_input(path: str | Path) -> np.ndarray:
    """Mono 16 kHz with the 50 Hz high-pass of the reference recipe, peak-normalised to 0.9."""
    x, _ = ffmpeg.decode(path, sr=SR_IN, channels=1, af="highpass=f=50")
    w = x[:, 0]
    return (0.9 * w / (np.abs(w).max() + 1e-9)).astype(np.float32)


class Sidon:
    name = "sidon"

    def __init__(self, device: str = "auto", models=None):
        """`models=(feature_extractor, decoder, preprocessor)` lets tests inject stand-ins."""
        self.device = pick_device(device) if models is None else "cpu"
        self.fe, self.dec, self.pre = models if models is not None else _load(self.device)

    def _features(self, xp: np.ndarray, ea: int, eb: int):
        import torch

        seg = xp[ea * HOP_IN: eb * HOP_IN]
        feats = self.pre(np.pad(seg, (160, 160)), sampling_rate=SR_IN, return_tensors="pt")["input_features"]
        if feats.shape[1] != eb - ea:
            raise RuntimeError(f"feature count {feats.shape[1]} != {eb - ea}")
        with torch.inference_mode():
            return self.fe(feats.to(self.device))["last_hidden_state"]

    def _decode(self, f, lo: int, hi: int) -> np.ndarray:
        """Samples of feature frames [lo, hi) of f, exactly (hi - lo) * 960 of them.

        The decoder keeps 48 kHz activations in memory (about 0.2 GB per second of audio on CPU),
        so it runs on short windows. It is local, so windows with a little context on each side
        give the same samples as one long call (checked to 1e-6 in tests).
        """
        import torch

        out = []
        n_feat = f.shape[1]
        with torch.inference_mode():
            for s in range(lo, hi, DEC_FRAMES):
                e = min(s + DEC_FRAMES, hi)
                cs, ce = max(0, s - DEC_CONTEXT), min(n_feat, e + DEC_CONTEXT)
                y = self.dec(f[:, cs:ce].transpose(1, 2)).view(-1).float().cpu().numpy()
                if len(y) != (ce - cs) * HOP_OUT - DECODER_TAIL_LOSS:
                    raise RuntimeError(f"unexpected decoder length {len(y)} for {ce - cs} frames")
                a0, a1 = (s - cs) * HOP_OUT, (e - cs) * HOP_OUT
                if a1 > len(y):
                    raise RuntimeError("decoder window without right context")
                out.append(y[a0:a1])
        return np.concatenate(out)

    def restore(self, x16: np.ndarray, out_path: str | Path, *, target: int | None = None,
                progress: Callable[[str], None] | None = None,
                chunk_frames: int = CHUNK_FRAMES, context_frames: int = CONTEXT_FRAMES) -> Path:
        """Restore a 16 kHz mono signal into a 48 kHz mono float WAV.

        `target` is the exact output length in 48 kHz samples (the source duration at 48 kHz);
        by default 3x the 16 kHz length.
        """
        n_in = len(x16)
        target = int(target) if target is not None else n_in * (SR_OUT // SR_IN)
        lat = LATENCY_48K
        need_frames = math.ceil((target + max(lat, 0)) / HOP_OUT) + 1
        total_frames = need_frames + context_frames + 1
        xp = np.zeros(total_frames * HOP_IN, dtype=np.float32)
        xp[:n_in] = x16
        n_chunks = math.ceil(need_frames / chunk_frames)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        skip = max(lat, 0)             # samples to drop at the start (output is late)
        lead = max(-lat, 0)            # zeros to add at the start (output is early)
        written = 0
        t0 = time.time()
        with sf.SoundFile(str(out_path), "w", SR_OUT, 1, subtype="FLOAT") as fh:
            def emit(block: np.ndarray):
                nonlocal skip, written
                if skip:
                    cut = min(skip, len(block))
                    block, skip = block[cut:], skip - cut
                block = block[: max(0, target - written)]
                if len(block):
                    fh.write(block)
                    written += len(block)

            if lead:
                emit(np.zeros(lead, dtype=np.float32))
            tail = None  # decoded samples just past the previous chunk, for the crossfade
            for i, a in enumerate(range(0, need_frames, chunk_frames)):
                memory.check_during_run()
                b = min(a + chunk_frames, need_frames)
                ea, eb = max(0, a - context_frames), b + context_frames
                f = self._features(xp, ea, eb)
                y = self._decode(f, a - ea, b - ea + 1)  # one frame more: the crossfade tail
                del f
                body = y[: (b - a) * HOP_OUT].copy()
                if tail is not None:
                    n = min(XFADE_OUT, len(tail), len(body))
                    w = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n, dtype=np.float32))
                    body[:n] = tail[:n] * (1 - w) + body[:n] * w
                tail = y[(b - a) * HOP_OUT: (b - a) * HOP_OUT + XFADE_OUT].copy()
                emit(body)
                if progress:
                    el = time.time() - t0
                    eta = el / (i + 1) * (n_chunks - i - 1)
                    progress(f"sidon chunk {i + 1}/{n_chunks}, {el:.0f} s elapsed, ~{eta:.0f} s left")
            if written < target:
                emit(np.zeros(target - written, dtype=np.float32))
        return out_path
