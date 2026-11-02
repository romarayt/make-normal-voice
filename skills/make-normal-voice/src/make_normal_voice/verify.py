"""`verify`: measure the deliverable against its input and against the target. No "sounds good".

Each gate returns pass / fail / unverified with the numbers behind it. "unverified" is used when a
measurement is not trustworthy (for example sync against a very echoey input) or a model is
missing: the report says so instead of pretending.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import align, asr, ffmpeg, loudness, match, metrics, speaker, vad
from . import thresholds as T
from .audio import load_channel, load_mono

SR = 16000


@dataclass
class Mapping:
    """How input time maps to output time: t_out = (t_in - trim_start) (no drift by default)."""
    offset: float = 0.0          # t_out = t_in * (1 + ppm*1e-6) + offset
    ppm: float = 0.0

    def out_t(self, t_in: float) -> float:
        return t_in * (1 + self.ppm * 1e-6) + self.offset


@dataclass
class Gate:
    name: str
    status: str
    summary: str
    details: dict = field(default_factory=dict)

    def to_dict(self):
        return {"name": self.name, "status": self.status, "summary": self.summary, "details": self.details}


def _windows_pairs(x_in: np.ndarray, x_out: np.ndarray, windows: list[tuple[float, float]], mp: Mapping):
    for a, b in windows:
        oa = mp.out_t(a)
        ia, ib = int(a * SR), int(b * SR)
        n = ib - ia
        o0 = int(round(oa * SR))
        if o0 < 0 or o0 + n > len(x_out) or ib > len(x_in):
            continue
        yield (a, b), x_in[ia:ib], x_out[o0:o0 + n]


def gate_quality(pairs, input_scores: list[dict] | None = None) -> tuple[Gate, Gate, list[dict]]:
    rows = []
    for k, ((a, b), wi, wo) in enumerate(pairs):
        si = input_scores[k] if input_scores and k < len(input_scores) else metrics.score(wi)
        so = metrics.score(wo)
        rows.append({"start": a, "end": b, "in": {k2: si[k2] for k2 in ("srmr", "sig", "bak", "ovrl")}, "out": so})
    if not rows:
        g = Gate("quality", "unverified", "no comparable speech windows")
        return g, Gate("not_worse", "unverified", "no comparable speech windows"), rows
    med = {k: float(np.median([r["out"][k] for r in rows])) for k in ("srmr", "sig", "bak", "ovrl")}
    fails = []
    # SRMR is reported but not gated here: it depends on the voice and delivery as much as on the
    # room (clean studio speech spans SRMR 4-17); echo is caught by "not worse" and DNSMOS instead
    if med["ovrl"] < T.TARGET_OVRL:
        fails.append(f"OVRL {med['ovrl']:.2f} < {T.TARGET_OVRL}")
    if med["bak"] < T.TARGET_BAK:
        fails.append(f"BAK {med['bak']:.2f} < {T.TARGET_BAK}")
    q = Gate("quality", "fail" if fails else "pass",
             ("below podcast-mic target: " + "; ".join(fails)) if fails else
             f"median OVRL {med['ovrl']:.2f}, BAK {med['bak']:.2f}, SRMR {med['srmr']:.1f}",
             {"median": med, "target": {"ovrl": T.TARGET_OVRL, "bak": T.TARGET_BAK}})
    worse = []
    for r in rows:
        gain = r["out"]["ovrl"] - r["in"]["ovrl"]
        if gain < -T.OVRL_NOT_WORSE_TOL:
            worse.append(f"{r['start']:.0f}-{r['end']:.0f} s OVRL {r['in']['ovrl']:.2f}→{r['out']['ovrl']:.2f}")
        # SRMR depends on the voice: for some voices room echo even raises it. A lower SRMR counts
        # only when DNSMOS does not confirm a clear improvement on the same window.
        if gain >= T.SRMR_OVERRIDE_OVRL_GAIN:
            continue
        if r["in"]["srmr"] >= T.DRY_INPUT_SRMR:
            if r["out"]["srmr"] < T.SRMR_FLOOR:
                worse.append(f"{r['start']:.0f}-{r['end']:.0f} s SRMR {r['out']['srmr']:.1f} < floor {T.SRMR_FLOOR}")
        elif r["out"]["srmr"] < r["in"]["srmr"]:
            worse.append(f"{r['start']:.0f}-{r['end']:.0f} s SRMR {r['in']['srmr']:.1f}→{r['out']['srmr']:.1f}")
    nw = Gate("not_worse", "fail" if worse else "pass", "; ".join(worse) if worse else
              f"no window got worse ({len(rows)} windows)", {"windows": rows})
    return q, nw, rows


def gate_words(pairs, model: str = "small", language: str | None = None, max_s: float = 30.0) -> Gate:
    """Words unchanged: confidently recognised input words must survive, beyond the recogniser's own noise."""
    if not asr.available(model):
        return Gate("words", "unverified", f"speech recogniser `whisper-{model}` not installed")
    rows = []
    for (a, b), wi, wo in pairs:
        n = int(min(max_s, b - a) * SR)
        wi, wo = wi[:n], wo[:n]
        ref_a, words_a, lang = asr.transcribe(wi, model=model, language=language, beam_size=5, words=True)
        ref_b, _, _ = asr.transcribe(wi, model=model, language=lang, beam_size=1)
        hyp, _, _ = asr.transcribe(wo, model=model, language=lang, beam_size=5)
        floor, n_conf, _ = asr.confident_loss(words_a, ref_b)
        loss, _, lost = asr.confident_loss(words_a, hyp)
        rows.append({"start": a, "end": a + n / SR, "language": lang, "confident_words": n_conf,
                     "floor": round(floor, 3), "lost": round(loss, 3), "lost_words": lost[:10],
                     "wer": round(asr.wer(ref_a, hyp), 3), "ok": bool(loss <= floor + T.WER_MARGIN),
                     "input": ref_a, "output": hyp})
    if not rows:
        return Gate("words", "unverified", "no comparable speech windows")
    bad = [r for r in rows if not r["ok"]]
    s = "; ".join(f"{r['start']:.0f} s: {r['lost']:.0%} of {r['confident_words']} confident words changed "
                  f"(recogniser noise {r['floor']:.0%})" for r in (bad or rows))
    return Gate("words", "fail" if bad else "pass", s, {"windows": rows, "margin": T.WER_MARGIN})


def gate_speaker(pairs) -> Gate:
    try:
        rows = [{"start": a, "end": b, "similarity": round(speaker.similarity(wi, wo), 3)} for (a, b), wi, wo in pairs]
    except Exception as e:  # noqa: BLE001 - missing model or runtime problem is reported, not hidden
        return Gate("speaker", "unverified", f"speaker check unavailable: {e}")
    if not rows:
        return Gate("speaker", "unverified", "no comparable speech windows")
    low = [r for r in rows if r["similarity"] < T.SPEAKER_MIN]
    lo = min(r["similarity"] for r in rows)
    return Gate("speaker", "fail" if low else "pass",
                f"lowest similarity {lo:.2f} (threshold {T.SPEAKER_MIN})", {"windows": rows})


def gate_sync(x_in: np.ndarray, x_out: np.ndarray, mp: Mapping, *, expected_len: int | None,
              out_len: int | None, input_srmr: float | None = None, points: int = T.SYNC_POINTS) -> Gate:
    """Offset of output against input on a 1 ms envelope grid at `points` places along the file."""
    dur_out = len(x_out) / SR
    rows = []
    for c in np.linspace(15.0, max(15.0, dur_out - 15.0), points):
        # where does output time c come from in the input?
        t_in = (c - mp.offset) / (1 + mp.ppm * 1e-6)
        if t_in < 10 or t_in > len(x_in) / SR - 10:
            continue
        o = align.local_offset(x_in, x_out, SR, c, win=20.0, search=0.06, guess=t_in - c)
        if not np.isfinite(o.seconds):
            continue
        # o.seconds: x_in[t + s] ~ x_out[t]; expected s = t_in - c
        err_ms = (o.seconds - (t_in - c)) * -1000.0  # positive = output late
        rows.append({"t_out": round(float(c), 1), "offset_ms": round(float(err_ms), 3), "corr": round(o.score, 3)})
    det = {"points": rows}
    len_ok = True
    if expected_len is not None and out_len is not None:
        det["length_samples"] = {"expected": int(expected_len), "got": int(out_len)}
        len_ok = int(expected_len) == int(out_len)
    if len(rows) < 3:
        return Gate("sync", "unverified" if len_ok else "fail", "too few speech points to measure sync", det)
    offs = np.array([r["offset_ms"] for r in rows])
    corr = np.array([r["corr"] for r in rows])
    med = float(np.median(offs))
    q1, q3 = np.percentile(offs, [25, 75])
    det.update({"median_ms": round(med, 3), "iqr_ms": round(float(q3 - q1), 3), "median_corr": round(float(np.median(corr)), 3)})
    reliable = np.median(corr) >= 0.65 and (q3 - q1) <= 1.5 and (input_srmr is None or input_srmr >= 7.0)
    det["reliable"] = bool(reliable)
    if not len_ok:
        return Gate("sync", "fail", f"length {out_len} samples, expected {expected_len}", det)
    if not reliable:
        # an echoey or noisy input smears its own envelope, so the offset measured against it is
        # biased by several ms; never call that a pass or a fail (and never "fix" by it)
        return Gate("sync", "unverified",
                    f"median offset {med:+.2f} ms, spread {q3 - q1:.1f} ms: the input is too echoey/noisy to measure "
                    "sync to 1 ms by envelope; length is exact", det)
    if abs(med) <= T.SYNC_MS:
        return Gate("sync", "pass", f"median offset {med:+.2f} ms over {len(rows)} points, length exact", det)
    return Gate("sync", "fail", f"median offset {med:+.2f} ms over {len(rows)} points", det)


def gate_loudness(out_path: Path, *, target: float = T.LOUDNESS_TARGET, segments: list[tuple[float, float]] | None = None) -> Gate:
    m = ffmpeg.loudness(out_path)
    fails = []
    if abs(m.integrated - target) > T.LOUDNESS_TOL:
        fails.append(f"integrated {m.integrated:.1f} LUFS (target {target:.1f} ± {T.LOUDNESS_TOL})")
    if m.true_peak > T.TRUE_PEAK_MAX:
        fails.append(f"true peak {m.true_peak:.1f} dBTP > {T.TRUE_PEAK_MAX}")
    x = load_channel(out_path, 48000)  # one channel of the dual-mono file; reported as stereo below
    dur = len(x) / 48000
    if not segments:
        n = max(1, int(dur // 300))
        edges = np.linspace(0, dur, n + 1)
        segments = list(zip(edges[:-1], edges[1:]))
    seg_rows = []
    for a, b in segments:
        if b - a < 30:
            continue
        v = loudness.integrated(x[int(a * 48000): int(b * 48000)], 48000, channels=2)
        seg_rows.append({"start": round(float(a), 1), "end": round(float(b), 1), "lufs": round(v, 2)})
        if abs(v - target) > T.SEGMENT_TOL:
            fails.append(f"{a / 60:.1f}-{b / 60:.1f} min at {v:.1f} LUFS")
    det = {"integrated": m.integrated, "true_peak": m.true_peak, "lra": m.lra, "segments": seg_rows}
    return Gate("loudness", "fail" if fails else "pass", "; ".join(fails) if fails else
                f"{m.integrated:.1f} LUFS, true peak {m.true_peak:.1f} dBTP, parts within ±{T.SEGMENT_TOL} LU", det)


def gate_timbre(out_path: Path, reference: Path | None) -> Gate:
    if reference is None:
        return Gate("timbre", "skipped", "no reference track given; timbre is reported, not gated", {})
    r = load_channel(reference, 48000)
    x = load_channel(out_path, 48000)
    cs, d = match.residual(r, x, 48000)
    worst = float(np.max(np.abs(d)))
    det = {"bands_hz": [round(float(c)) for c in cs], "diff_db": [round(float(v), 2) for v in d]}
    return Gate("timbre", "pass" if worst <= T.TIMBRE_REF_DB else "fail",
                f"largest band difference to the reference {worst:.2f} dB (limit ±{T.TIMBRE_REF_DB})", det)


def seam_spikes(x, sr: int, seams: list[float], *, n_baseline: int = 400, seed: int = 0) -> dict:
    """A click at a seam = a sample-level discontinuity exactly there: the second difference within
    +-3 samples of the seam, against its RMS over +-10 ms. Speech, even a fricative onset, is smooth
    at that scale; a glued step or impulse is not. The limit adapts to the file (random places).

    `x` is an array or a WAV path (then only the few milliseconds around each place are read)."""
    import soundfile as sf

    w = int(0.01 * sr)
    if isinstance(x, (str, Path)):
        fh = sf.SoundFile(str(x))
        total = fh.frames

        def read(k0, k1):
            fh.seek(k0)
            return fh.read(k1 - k0, dtype="float64", always_2d=True)[:, 0]
    else:
        fh, total = None, len(x)

        def read(k0, k1):
            return np.asarray(x[k0:k1], dtype=np.float64)

    def ratio(t):
        k = int(round(t * sr)) - 1
        if k - w - 1 < 0 or k + w + 2 >= total:
            return 0.0
        d2 = np.abs(np.diff(read(k - w - 1, k + w + 2), 2))  # d2[i] belongs to sample k - w + i
        c = w
        around = np.concatenate([d2[c - w: c - 4], d2[c + 5: c + w]])
        peak = float(d2[c - 3: c + 4].max())
        if peak < 1e-3:  # a step below -60 dBFS: inaudible
            return 0.0
        return peak / (float(np.sqrt(np.mean(around ** 2))) + 1e-12)

    rng = np.random.default_rng(seed)
    dur = total / sr
    base = np.array([ratio(t) for t in rng.uniform(0.05, max(0.06, dur - 0.05), n_baseline)])
    limit = max(10.0, float(np.percentile(base, 99.5))) if len(base) else 10.0
    rows = [{"seam_s": round(float(t), 4), "ratio": round(ratio(t), 2)} for t in seams if 0.05 < t < dur - 0.05]
    if fh is not None:
        fh.close()
    return {"limit": round(limit, 2), "seams": rows, "bad": [r for r in rows if r["ratio"] > limit]}


def _pause_offsets_db(x_out: np.ndarray, x_in: np.ndarray, segs_in: list[tuple[float, float]], mp: Mapping,
                      a: float, b: float) -> np.ndarray:
    """Level of the output minus level of the input, in dB, frame by frame (50 ms) over the input's
    pauses between input times a and b. Both levels are floored at -70 dBFS (silence)."""
    hop = int(0.05 * SR)
    vals = []
    for pa, pb in vad.pauses(segs_in, len(x_in) / SR, 0.15):
        lo, hi = max(pa, a), min(pb, b)
        for t in np.arange(lo, hi - 0.05, 0.05):
            fi = x_in[int(t * SR): int(t * SR) + hop]
            to = mp.out_t(t)
            if to < 0:
                continue
            fo = x_out[int(to * SR): int(to * SR) + hop]
            if len(fi) < hop or len(fo) < hop:
                continue
            li = max(-70.0, 10 * np.log10(np.mean(fi.astype(np.float64) ** 2) + 1e-12))
            lo_ = max(-70.0, 10 * np.log10(np.mean(fo.astype(np.float64) ** 2) + 1e-12))
            vals.append(lo_ - li)
    return np.array(vals)


def gate_artifacts(out_path: Path, mp: Mapping, x_in: np.ndarray, x_out: np.ndarray, segs_in: list[tuple[float, float]],
                   seams: list[float] | None = None) -> Gate:
    """Seams and lost speech. Levels are read from the 16 kHz copies already in memory; clicks
    from the 48 kHz file itself, a few milliseconds at a time."""
    fails = []
    dur_out = len(x_out) / SR
    seams = [t for t in (seams or []) if 0.0 < t < dur_out]  # a source may start before the timeline
    sp = seam_spikes(out_path, 48000, seams)
    if sp["bad"]:
        fails.append(f"click at {len(sp['bad'])} seam(s) (first at {sp['bad'][0]['seam_s']:.2f} s)")
    # Background jump at a seam: how the output's pauses relate to the input's, chunk by chunk.
    # Each side spans the neighbouring chunk (at most 30 s); the same input pauses are used for both
    # signals, so what the restoration does to pauses everywhere cancels out.
    jumps, steps = [], []
    edges = sorted(seams)
    inv = lambda t: (t - mp.offset) / (1 + mp.ppm * 1e-6)  # noqa: E731 - output time -> input time
    for i, s in enumerate(edges):
        left = max(edges[i - 1] if i else 0.0, s - 30.0)
        right = min(edges[i + 1] if i + 1 < len(edges) else dur_out, s + 30.0)
        dl = _pause_offsets_db(x_out, x_in, segs_in, mp, inv(left), inv(s))
        dr = _pause_offsets_db(x_out, x_in, segs_in, mp, inv(s), inv(right))
        if len(dl) >= 20 and len(dr) >= 20:  # at least 1 s of pauses on each side
            steps.append((s, float(np.median(dr) - np.median(dl))))
    mags = np.array([abs(v) for _, v in steps])
    jump_limit = 10.0
    if len(mags) >= 5:
        med = float(np.median(mags))
        mad = float(np.median(np.abs(mags - med))) + 1e-6
        jump_limit = max(jump_limit, med + 5 * 1.4826 * mad)
    for s, v in steps:
        if abs(v) > jump_limit:
            jumps.append({"seam_s": round(s, 2), "step_db": round(v, 1)})
    if jumps:
        fails.append(f"background level jumps at {len(jumps)} seam(s) that the input does not have")
    holes = []
    for a, b in segs_in:
        if b - a < 0.8:
            continue
        oa = mp.out_t(a)
        seg_o = x_out[int(oa * SR): int((oa + (b - a)) * SR)]
        if len(seg_o) < 400:
            continue
        eo = 10 * np.log10(np.mean(seg_o.astype(np.float64) ** 2) + 1e-12)
        if eo < -60:
            holes.append({"t_out": round(oa, 2), "dur": round(b - a, 2), "level_db": round(float(eo), 1)})
    if holes:
        fails.append(f"{len(holes)} stretch(es) where the input has speech but the output is silent")
    det = {"seam_clicks": sp, "seam_jumps": jumps, "jump_limit_db": round(jump_limit, 1), "silent_speech": holes[:20],
           "seams_checked": len(seams or [])}
    return Gate("artifacts", "fail" if fails else "pass", "; ".join(fails) if fails else
                f"no clicks or background jumps at {len(seams or [])} seam(s), no lost speech", det)


def verify(in_path: str | Path, out_path: str | Path, *, windows: list[tuple[float, float]] | None = None,
           input_scores: list[dict] | None = None, mapping: Mapping | None = None, reference: str | Path | None = None,
           expected_len: int | None = None, target: float = T.LOUDNESS_TARGET, asr_model: str = "small",
           language: str | None = None, seams: list[float] | None = None, segments: list[tuple[float, float]] | None = None,
           skip: tuple[str, ...] = ()) -> dict:
    in_path, out_path = Path(in_path), Path(out_path)
    mp = mapping or Mapping()
    x_in = load_mono(in_path, SR)
    x_out = load_mono(out_path, SR)
    segs_in = vad.speech_segments(x_in)
    if windows is None:
        windows = vad.speech_windows(segs_in, len(x_in) / SR, 60.0, 3)
    pairs = list(_windows_pairs(x_in, x_out, windows, mp))
    gates: list[Gate] = []
    q, nw, _ = gate_quality(pairs, input_scores)
    gates += [q, nw]
    if "words" not in skip:
        gates.append(gate_words(pairs, asr_model, language))
    if "speaker" not in skip:
        gates.append(gate_speaker(pairs))
    info = ffmpeg.audio_streams(ffmpeg.probe(out_path))
    out_len = int(round(float(info[0].get("duration", 0)) * 48000)) if info else None
    try:
        import soundfile as sf
        out_len = sf.info(str(out_path)).frames
    except Exception:  # noqa: BLE001
        pass
    in_srmr = float(np.median([r["srmr"] for r in input_scores])) if input_scores else None
    gates.append(gate_sync(x_in, x_out, mp, expected_len=expected_len, out_len=out_len, input_srmr=in_srmr))
    gates.append(gate_loudness(out_path, target=target, segments=segments))
    gates.append(gate_timbre(out_path, Path(reference) if reference else None))
    gates.append(gate_artifacts(out_path, mp, x_in, x_out, segs_in, seams))
    ok = all(g.status in ("pass", "skipped", "unverified") for g in gates)
    return {"ok": ok, "gates": [g.to_dict() for g in gates]}
