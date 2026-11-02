"""Processing stages that the pipeline strings together. Each one keeps the timeline intact."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np
import soundfile as sf

from . import __version__, ffmpeg, memory, paths, vad
from .audio import load_mono, write_wav
from .engines import sidon as sidon_engine

SR48 = 48000
last_chunk_s = 30.0  # chunk length the last Sidon run used (memory may shorten the requested one)


def source_len48(path: Path) -> int:
    """Length of the source at 48 kHz in samples: what the deliverable must match exactly."""
    info = ffmpeg.audio_streams(ffmpeg.probe(path))[0]
    sr = int(info["sample_rate"])
    try:
        n = sf.info(str(path)).frames
    except Exception:  # noqa: BLE001 - not a soundfile format (mp4, m4a ...): count decoded samples
        n = sum(len(b) for b in ffmpeg.stream_blocks(path, sr, 1, block_s=120.0))
    return int(round(n * SR48 / sr))


def _hash_array(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:24]


def mono_source(path: Path, channel: int | None, pre_af: str | None = None) -> Path:
    """The input as one 48 kHz float mono WAV (picked channel or downmix), for master-only routes."""
    out = paths.cache_dir() / "mono" / (hashlib.sha256(f"{path}|{path.stat().st_mtime}|{channel}|{pre_af}".encode()).hexdigest()[:24] + ".wav")
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    af = []
    if channel is not None:
        af.append(f"pan=mono|c0=c{channel}")
    if pre_af:
        af.append(pre_af)
    args = ["-v", "error", "-y", "-i", str(path), "-map", "0:a:0", "-vn"]
    if af:
        args += ["-af", ",".join(af)]
    args += ["-ac", "1", "-ar", str(SR48), "-c:a", "pcm_f32le", str(out)]
    ffmpeg.run(args)
    return out


def sidon_restore(path: Path, *, target_len: int, channel: int | None = None, pre_af: str | None = None,
                  device: str = "auto", log=print, start: float | None = None, duration: float | None = None,
                  chunk_s: float | None = None) -> Path:
    """Run Sidon with a cache keyed by the exact decoded input, so a re-run takes minutes, not a quarter hour."""
    af = []
    if channel is not None:
        af.append(f"pan=mono|c0=c{channel}")
    if pre_af:
        af.append(pre_af)
    af.append("highpass=f=50")
    x, _ = ffmpeg.decode(path, sr=16000, channels=1, af=",".join(af), start=start, duration=duration)
    x = x[:, 0]
    x = (0.9 * x / (np.abs(x).max() + 1e-9)).astype(np.float32)
    global last_chunk_s
    chunk_s = memory.pick_chunk_seconds(chunk_s)
    last_chunk_s = chunk_s
    chunk_frames = max(500, int(round(chunk_s * 50)))
    key = f"{_hash_array(x)}-{target_len}-{sidon_engine.LATENCY_48K}-{chunk_frames}-{__version__}"
    out = paths.cache_dir() / "sidon" / f"{key}.wav"
    if out.exists() and sf.info(str(out)).frames == target_len:
        log(f"Sidon: using cached result {out.name}")
        return out
    need = target_len * 4 * 2 + 1e9
    if paths.free_bytes(out.parent) < need:
        raise RuntimeError(f"not enough disk for Sidon output: need ~{need / 1e9:.1f} GB in {out.parent}")
    tmp = out.with_suffix(".part.wav")
    inp = out.with_suffix(".in.npy")
    np.save(inp, x)
    log(f"Sidon: {len(x) / 16000 / 60:.1f} min of audio in {chunk_s:.0f} s chunks")
    # a separate process, so the model's several GB are returned to the system before the mix
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=_sidon_worker, args=(str(inp), str(tmp), target_len, device, chunk_frames))
    p.start()
    p.join()
    inp.unlink(missing_ok=True)
    if p.exitcode != 0 or not tmp.exists():
        raise RuntimeError(f"Sidon worker failed (exit code {p.exitcode})")
    tmp.replace(out)
    meta = {"source": str(path), "frames": target_len, "version": __version__}
    out.with_suffix(".json").write_text(json.dumps(meta))
    return out


def _sidon_worker(inp: str, out: str, target_len: int, device: str, chunk_frames: int) -> None:
    x = np.load(inp)
    eng = sidon_engine.Sidon(device)
    print(f"Sidon on {eng.device}", flush=True)
    eng.restore(x, out, target=target_len, progress=lambda m: print(m, flush=True), chunk_frames=chunk_frames)


def seams(duration: float, chunk_s: float = 30.0) -> list[float]:
    """Where Sidon chunks meet in the output timeline (seconds)."""
    step = max(500, int(round(chunk_s * 50))) * sidon_engine.HOP_OUT / SR48
    return [float(t) for t in np.arange(step, duration, step)]


def level_parts(x: np.ndarray, sr: int, parts: list[dict], ramp: float = 0.05) -> tuple[np.ndarray, list[dict]]:
    """Bring parts of different loudness to the loudness of the longest part, switching in pauses."""
    from . import loudness

    x16 = load_mono_array(x, sr)
    segs = vad.speech_segments(x16)
    mask = vad.speech_mask(segs, len(x16), 16000)
    lv = []
    for p in parts:
        a, b = int(p["start"] * 16000), int(p["end"] * 16000)
        seg = x16[a:b][mask[a:b]]
        lv.append(loudness.integrated(seg, 16000) if len(seg) > 16000 else np.nan)
    lengths = [(p["end"] - p["start"]) if np.isfinite(v) else -1.0 for p, v in zip(parts, lv)]
    if max(lengths) <= 0:
        return x, [{"start": p["start"], "end": p["end"], "gain_db": 0.0, "note": "no speech to measure"} for p in parts]
    ref = lv[int(np.argmax(lengths))]
    gains = [0.0 if not np.isfinite(v) else float(ref - v) for v in lv]
    g = np.zeros(len(x), dtype=np.float32)
    for p, gd in zip(parts, gains):
        g[int(p["start"] * sr): int(p["end"] * sr)] = gd
    # smooth the gain steps (they sit in pauses) over `ramp` seconds
    n = max(1, int(ramp * sr))
    k = np.ones(n, dtype=np.float32) / n
    gs = np.convolve(np.pad(g, (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid")
    y = (x * (10 ** (gs / 20))).astype(np.float32)
    log = [{"start": p["start"], "end": p["end"], "speech_lufs_after_restore": None if not np.isfinite(v) else round(v, 2),
            "gain_db": round(gd, 2)} for p, v, gd in zip(parts, lv, gains)]
    return y, log


def load_mono_array(x: np.ndarray, sr: int) -> np.ndarray:
    import soxr

    return soxr.resample(x, sr, 16000, quality="HQ").astype(np.float32) if sr != 16000 else x


def read48(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SR48:
        return load_mono(path, SR48)
    return x[:, 0]


def write48(path: Path, x: np.ndarray) -> Path:
    return write_wav(path, x, SR48, subtype="FLOAT")
