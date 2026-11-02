"""`analyze`: a passport of one recording, and the facts the router decides on.

Every number here is measured, not assumed. Quality (SRMR, DNSMOS) is measured on speech windows
picked by VAD at the start, middle and end, never on pauses.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal

from . import ffmpeg, loudness, metrics, vad
from .audio import load_mono

SR = 16000


def _file_info(path: Path) -> dict:
    info = ffmpeg.probe(path)
    streams = ffmpeg.audio_streams(info)
    fmt = info.get("format", {})
    out = {
        "path": str(path),
        "container": fmt.get("format_name"),
        "audio_streams": len(streams),
        "video": any(s.get("codec_type") == "video" for s in info.get("streams", [])),
        "probe_errors": info.get("_errors", [])[:10],
        "readable": bool(streams),
    }
    if streams:
        s = streams[0]
        out.update({
            "codec": s.get("codec_name"),
            "sample_rate": int(s.get("sample_rate", 0)),
            "channels": int(s.get("channels", 0)),
            "bits": int(s.get("bits_per_raw_sample") or s.get("bits_per_sample") or 0) or None,
            "sample_fmt": s.get("sample_fmt"),
            "duration": float(s.get("duration") or fmt.get("duration") or 0.0),
            "bitrate": int(s.get("bit_rate") or fmt.get("bit_rate") or 0) or None,
        })
    return out


def _damage_and_stereo(path: Path, sr: int, ch: int) -> dict:
    """One streaming pass at the native rate: clipping, zeros, clicks, DC, channel relations."""
    n = 0
    clip = flat = 0
    peak = 0.0
    dc = np.zeros(ch)
    zero_runs: list[tuple[float, float]] = []
    clicks = 0
    sxx = np.zeros(ch)
    sxy = 0.0
    prev_tail = np.zeros((0, ch), dtype=np.float32)
    for blk in ffmpeg.stream_blocks(path, sr, ch, block_s=30.0):
        a = np.abs(blk)
        peak = max(peak, float(a.max()) if len(a) else 0.0)
        clip += int((a >= 0.999).sum())
        dc += blk.sum(0)
        m = blk.mean(1)
        # flat tops: 3+ equal consecutive samples near full scale
        hi = np.abs(m) > 0.9
        eq = np.concatenate([[False], np.diff(m) == 0]) & hi
        flat += int((eq[1:] & eq[:-1]).sum())
        # digital zeros (exact zero on every channel) longer than 10 ms
        z = np.all(blk == 0, axis=1)
        idx = np.flatnonzero(np.diff(np.concatenate([[0], z.astype(np.int8), [0]])))
        for s, e in zip(idx[::2], idx[1::2]):
            if e - s >= 0.01 * sr:
                zero_runs.append(((n + s) / sr, (e - s) / sr))
        # clicks: second difference far above its local median absolute deviation
        x = np.concatenate([prev_tail[:, 0], blk[:, 0]]) if len(prev_tail) else blk[:, 0]
        d2 = np.abs(np.diff(x, 2))
        if len(d2) > 1000:
            med = signal.medfilt(d2[:: 16], 63)
            thr = 12 * np.repeat(med, 16)[: len(d2)] + 1e-4
            hits = np.flatnonzero(d2 > thr)
            if len(hits):
                clicks += int(1 + np.count_nonzero(np.diff(hits) > sr // 100))
        prev_tail = blk[-2:]
        if ch >= 2:
            sxx += (blk[:, :2].astype(np.float64) ** 2).sum(0)
            sxy += float((blk[:, 0].astype(np.float64) * blk[:, 1]).sum())
        n += len(blk)
    dur = n / sr if sr else 0.0
    out = {
        "peak_dbfs": round(20 * np.log10(peak + 1e-12), 2),
        "clip_ratio": clip / max(n * ch, 1),
        "flat_top_samples": flat,
        "dc_offset": [round(float(v / max(n, 1)), 6) for v in dc],
        "digital_zero_runs": [{"t": round(t, 3), "dur": round(d, 3)} for t, d in zero_runs[:50]],
        "digital_zero_count": len(zero_runs),
        "clicks_per_min": round(clicks / max(dur / 60, 1e-9), 2),
    }
    if ch >= 2:
        corr = sxy / (np.sqrt(sxx[0] * sxx[1]) + 1e-20)
        lvl = 10 * np.log10((sxx + 1e-20) / max(n, 1))
        out["stereo"] = {
            "correlation": round(float(corr), 4),
            "level_db": [round(float(v), 1) for v in lvl],
            "dead_channel": int(np.argmin(lvl)) if abs(lvl[0] - lvl[1]) > 30 or min(lvl) < -90 else None,
            "inverted": bool(corr < -0.8),
            "identical": bool(corr > 0.999 and abs(lvl[0] - lvl[1]) < 0.1),
        }
    return out


def _band(path: Path, sr: int, windows: list[tuple[float, float]]) -> dict:
    """Where the speech spectrum ends: phone (~4 kHz), 8 kHz codecs, 16 kHz codecs, or full band."""
    if not windows:
        return {"cutoff_hz": None, "class": "unknown"}
    a, b = windows[len(windows) // 2]
    x, _ = ffmpeg.decode(path, sr=sr, channels=1, start=a, duration=min(60.0, b - a))
    f, p = signal.welch(x[:, 0], fs=sr, nperseg=4096)
    pdb = 10 * np.log10(p + 1e-20)
    ref = np.median(pdb[(f > 300) & (f < 3000)])
    floor = np.percentile(pdb, 2)
    above = np.flatnonzero(pdb > max(ref - 50, floor + 10))
    cutoff = float(f[above[-1]]) if len(above) else 0.0
    nyq = sr / 2
    cutoff = min(cutoff, nyq)
    cls = "full"
    if cutoff < 4600:
        cls = "phone"
    elif cutoff < 8600:
        cls = "narrow"
    elif cutoff < 16500 and nyq > 17000:
        cls = "codec"
    return {"cutoff_hz": round(cutoff), "class": cls}


def _hum(x16: np.ndarray, pauses: list[tuple[float, float]]) -> dict:
    """Mains hum: prominence of 50/60 Hz and harmonics in the pauses (or whole signal)."""
    parts = [x16[int(a * SR): int(b * SR)] for a, b in pauses if b - a > 0.5][:200]
    sig = np.concatenate(parts) if parts and sum(map(len, parts)) > SR * 4 else x16
    if len(sig) < SR * 2:
        return {"freq": None, "prominence_db": 0.0}
    f, p = signal.welch(sig, fs=SR, nperseg=16384)
    pdb = 10 * np.log10(p + 1e-20)
    best = (None, 0.0)
    for base in (50, 60):
        proms = []
        for h in range(1, 5):
            fc = base * h
            i = int(np.argmin(np.abs(f - fc)))
            nb = pdb[(np.abs(f - fc) > 3) & (np.abs(f - fc) < 12)]
            proms.append(float(pdb[max(0, i - 1): i + 2].max() - np.median(nb)))
        score = float(np.mean(sorted(proms)[-2:]))
        if score > best[1]:
            best = (base, score)
    return {"freq": best[0] if best[1] > 10 else None, "prominence_db": round(best[1], 1)}


def _parts(x16: np.ndarray, segs: list[tuple[float, float]], win: float = 10.0, min_part: float = 60.0) -> list[dict]:
    """Stretches of different speech loudness (e.g. a part recorded before the mic was switched on)."""
    dur = len(x16) / SR
    mask = vad.speech_mask(segs, len(x16), SR)
    vals = []
    for t in np.arange(0, dur, win):
        a, b = int(t * SR), int(min(dur, t + win) * SR)
        seg = x16[a:b][mask[a:b]]
        vals.append(loudness.integrated(seg, SR) if len(seg) > SR else np.nan)
    v = np.array(vals)
    t = np.arange(len(v)) * win

    def split(lo: int, hi: int, depth: int) -> list[int]:
        k = int(min_part / win)
        best, bi = 0.0, None
        for i in range(lo + k, hi - k + 1):
            left, right = v[lo:i], v[i:hi]
            left, right = left[np.isfinite(left)], right[np.isfinite(right)]
            if len(left) < k // 2 or len(right) < k // 2:
                continue
            d = abs(np.median(left) - np.median(right))
            spread = np.sqrt((np.var(left) / len(left)) + (np.var(right) / len(right)))
            if d > 1.0 and d > 3 * spread and d > best:
                best, bi = d, i
        if bi is None or depth >= 3:
            return []
        return split(lo, bi, depth + 1) + [bi] + split(bi, hi, depth + 1)

    cuts = split(0, len(v), 0) if np.isfinite(v).sum() > 2 * min_part / win else []
    bounds = [0.0] + [vad.nearest_pause(segs, dur, t[c], radius=win) for c in cuts] + [dur]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = x16[int(a * SR): int(b * SR)][mask[int(a * SR): int(b * SR)]]
        out.append({"start": round(a, 2), "end": round(b, 2), "speech_lufs": round(loudness.integrated(seg, SR), 2)})
    return out


def analyze(path: str | Path, *, quick: bool = False) -> dict:
    path = Path(path)
    p = {"file": _file_info(path)}
    fi = p["file"]
    if not fi["readable"]:
        p["verdict"] = "unreadable"
        return p
    errs = ffmpeg.decode_errors(path)
    fi["decode_errors"] = errs[:10]
    sr, ch = fi["sample_rate"], max(1, fi["channels"])
    x16 = load_mono(path, SR)
    dur = len(x16) / SR
    fi["decoded_duration"] = round(dur, 4)
    segs = vad.speech_segments(x16)
    p["speech"] = {"ratio": round(vad.speech_ratio(segs, dur), 3), "segments": len(segs),
                   "first": round(segs[0][0], 2) if segs else None, "last": round(segs[-1][1], 2) if segs else None}
    lv = ffmpeg.loudness(path)
    per_min = loudness.windows(x16, SR, 60.0)
    p["level"] = {"integrated": lv.integrated, "true_peak": lv.true_peak, "lra": lv.lra,
                  "per_minute": [[round(float(a), 1), round(float(b), 1)] for a, b in per_min],
                  "parts": _parts(x16, segs)}
    p["damage"] = _damage_and_stereo(path, sr, ch)
    if "stereo" in p["damage"]:
        p["stereo"] = p["damage"].pop("stereo")
    pz = vad.pauses(segs, dur)
    mask = vad.speech_mask(segs, len(x16), SR)
    sp = x16[mask]
    ps = x16[~mask]
    sp_db = 10 * np.log10(np.mean(sp.astype(np.float64) ** 2) + 1e-12) if len(sp) else -120.0
    ps_db = 10 * np.log10(np.percentile(ps.astype(np.float64) ** 2, 50) + 1e-12) if len(ps) > SR else -120.0
    p["noise"] = {"speech_db": round(float(sp_db), 1), "pause_db": round(float(ps_db), 1),
                  "snr_db": round(float(sp_db - ps_db), 1), "hum": _hum(x16, pz)}
    wins = vad.speech_windows(segs, dur, win=60.0, n=3)
    p["windows"] = []
    for a, b in wins:
        w = x16[int(a * SR): int(b * SR)]
        m = metrics.score(w, with_srmr=not quick)
        m.update({"start": round(a, 2), "end": round(b, 2)})
        p["windows"].append(m)
    p["band"] = _band(path, sr, wins)
    p["content"] = _content(x16, segs, pz, p["noise"])
    return p


def _content(x16: np.ndarray, segs, pz, noise: dict) -> dict:
    """Heuristics for music or other sounds under the voice. Flagged, not trusted blindly."""
    tonal = []
    for a, b in pz[:300]:
        if b - a < 0.6:
            continue
        seg = x16[int(a * SR): int(b * SR)]
        if 10 * np.log10(np.mean(seg.astype(np.float64) ** 2) + 1e-12) < noise["speech_db"] - 35:
            continue
        f, pw = signal.welch(seg, fs=SR, nperseg=2048)
        band = (f > 100) & (f < 4000)
        flat = np.exp(np.mean(np.log(pw[band] + 1e-20))) / (np.mean(pw[band]) + 1e-20)
        tonal.append(flat)
    loud_pauses = len(tonal)
    music = bool(loud_pauses >= 5 and np.median(tonal) < 0.05 and noise["snr_db"] < 25)
    return {"music_suspected": music, "loud_tonal_pauses": loud_pauses,
            "overlap": "not measured (no overlap detector in v0.1)"}


def summary(p: dict) -> dict:
    """Median quality numbers across the speech windows."""
    w = p.get("windows", [])
    if not w:
        return {}
    keys = [k for k in ("srmr", "sig", "bak", "ovrl") if k in w[0]]
    return {k: round(float(np.median([x[k] for x in w])), 3) for k in keys}
