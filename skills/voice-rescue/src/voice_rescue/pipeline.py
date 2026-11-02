"""`process`: analyse → explain the route → probe the hardest stretch → full run → verify → fix (≤3 rounds).

The deliverable keeps the source timeline: same length, same start, unless the user asked to trim
or to put the track on another file's timeline (`sync_to`, e.g. the video).
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import align, asr, match, paths, report, stages, vad, verify
from . import analyze as analyze_mod
from . import master as master_mod
from . import thresholds as T
from .audio import load_mono, write_wav
from .route import plan as make_plan

SR48 = 48000


@dataclass
class Options:
    out_dir: Path
    reference: Path | None = None
    sync_to: Path | None = None
    trim_start: float | None = None
    trim_end: float | None = None
    trim_tail: bool = False
    target: float = T.LOUDNESS_TARGET
    device: str = "auto"
    asr_model: str = "small"
    language: str | None = None
    probe: bool = True
    lang: str = "en"
    keep_work: bool = False
    experimental: dict = field(default_factory=dict)
    force_route: str | None = None   # "sidon" or "master"
    name: str | None = None
    chunk_s: float = 30.0            # Sidon chunk length; shorter needs less memory (see memory.py)
    log: object = None


class Run:
    def __init__(self, inputs: list[Path], opts: Options):
        self.inputs = [Path(p) for p in inputs]
        self.o = opts
        self.out = Path(opts.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.work = self.out / "_work"
        self.work.mkdir(exist_ok=True)
        self.log = opts.log or (lambda m: print(m, flush=True))
        self.events: list[str] = []
        self.t0 = time.time()

    def note(self, msg: str):
        self.events.append(msg)
        self.log(msg)

    # ------------------------------------------------------------------ helpers
    def _disk_check(self, seconds: float):
        need = seconds * SR48 * (4 * 4 + 3 * 2 * 2) + 1.5e9
        free = paths.free_bytes(self.out)
        free_cache = paths.free_bytes(paths.cache_dir())
        if free < need or free_cache < seconds * SR48 * 4 + 5e8:
            raise RuntimeError(f"not enough free disk: need about {need / 1e9:.1f} GB, have {free / 1e9:.1f} GB at {self.out}")

    def _restore_track(self, src: Path, pl, *, start=None, duration=None, target_len=None) -> np.ndarray:
        """Mono 48 kHz restored (or cleaned-only) signal of `src` on its own timeline."""
        channel = pl.params.get("channel")
        pre = []
        if "declip" in pl.steps:
            pre.append("adeclip")
        if "hum_notch" in pl.steps:
            f0 = pl.params["hum_freq"]
            pre.append(",".join(f"bandreject=f={f0 * h}:width_type=q:w=30" for h in range(1, 5)))
        pre_af = ",".join(pre) or None
        if target_len is None:
            target_len = stages.source_len48(src) if start is None else int(round(duration * SR48))
        if "sidon" in pl.steps:
            p = stages.sidon_restore(src, target_len=target_len, channel=channel, pre_af=pre_af, device=self.o.device,
                                     log=self.log, start=start, duration=duration, chunk_s=self.o.chunk_s)
            x = stages.read48(p)
        else:
            if start is None:
                x = stages.read48(stages.mono_source(src, channel, pre_af))
            else:
                x = load_mono(src, SR48, start=start, duration=duration,
                              af=",".join(a for a in ([f"pan=mono|c0=c{channel}"] if channel is not None else []) + ([pre_af] if pre_af else [])) or None)
        if len(x) < target_len:
            x = np.pad(x, (0, target_len - len(x)))
        return x[:target_len]

    def _finish(self, x: np.ndarray, pl, tag: str, *, reference: np.ndarray | None) -> tuple[Path, dict]:
        info: dict = {}
        if "level_parts" in pl.steps and pl.params.get("parts"):
            x, info["level_parts"] = stages.level_parts(x, SR48, pl.params["parts"])
        if not np.all(np.isfinite(x)):
            raise RuntimeError(f"internal error: {np.count_nonzero(~np.isfinite(x))} non-finite samples before the mix "
                               f"({tag}); nothing was delivered")
        pre = stages.write48(self.work / f"{tag}_premaster.wav", x)
        out = self.work / f"{tag}_master.wav"
        info["master"] = master_mod.master(pre, out, target=self.o.target).to_dict()
        if reference is not None:
            # the reference is a finished track: compare it with our *mixed* result (the mix's
            # high-pass and de-esser shape the spectrum too), correct before the mix, mix again
            info["timbre_match"] = []
            for _ in range(2):
                cs, d = match.correction(reference, stages.read48(out), SR48)
                x = match.apply_fir(x, match.design_fir(cs, d, SR48))
                info["timbre_match"].append({"bands_hz": [round(float(c)) for c in cs],
                                             "gain_db": [round(float(v), 2) for v in d]})
                stages.write48(pre, x)
                info["master"] = master_mod.master(pre, out, target=self.o.target).to_dict()
        return out, info

    # ------------------------------------------------------------------ main
    def process(self) -> dict:
        if len(self.inputs) > 1 or self.o.sync_to:
            return self._process_session()
        return self._process_single(self.inputs[0])

    def _probe(self, src: Path, passport: dict, pl, reference) -> dict | None:
        dur = passport["file"].get("decoded_duration", 0)
        if not self.o.probe or dur < T.PROBE_MIN_DURATION or not passport.get("windows"):
            return None
        worst = min(passport["windows"], key=lambda w: w["ovrl"])
        c = (worst["start"] + worst["end"]) / 2
        a = max(0.0, min(dur - T.PROBE_LEN, c - T.PROBE_LEN / 2))
        self.note(f"Probe: {T.PROBE_LEN:.0f} s from {a / 60:.1f} min (hardest window, OVRL {worst['ovrl']:.2f}) before the full run")
        x = self._restore_track(src, pl, start=a, duration=T.PROBE_LEN)
        ppl = pl
        if pl.params.get("parts"):
            # parts are on the full timeline: move them into the probe window, drop what falls outside
            parts = [{**p, "start": max(0.0, p["start"] - a), "end": min(T.PROBE_LEN, p["end"] - a)}
                     for p in pl.params["parts"] if p["end"] > a and p["start"] < a + T.PROBE_LEN]
            ppl = type(pl)(**{**pl.__dict__, "params": {**pl.params, "parts": parts}})
            if len(parts) < 2:
                ppl.steps = [s for s in pl.steps if s != "level_parts"]
        out, _ = self._finish(x, ppl, "probe", reference=reference)
        pdir = self.out / "probe"
        pdir.mkdir(exist_ok=True)
        tgt = pdir / "probe_after.wav"
        shutil.copyfile(out, tgt)
        before = pdir / "probe_before.wav"
        report.loudness_matched_copy(src, before, start=a, duration=T.PROBE_LEN, like=tgt)
        win = [(a + 15, a + T.PROBE_LEN - 15)]
        v = verify.verify(before, tgt, windows=[(15.0, T.PROBE_LEN - 15)], mapping=verify.Mapping(),
                          asr_model=self.o.asr_model, language=self.o.language, skip=())
        res = {g["name"]: g["status"] for g in v["gates"]}
        self.note("Probe gates: " + ", ".join(f"{k} {s}" for k, s in res.items()))
        return {"start": a, "duration": T.PROBE_LEN, "window": win, "gates": v["gates"], "files": [str(before), str(tgt)]}

    def _process_single(self, src: Path) -> dict:
        self.note(f"Analysing {src.name}")
        pp = analyze_mod.analyze(src)
        if not pp["file"].get("readable"):
            return self._refuse(src, pp)
        pl = make_plan(pp, name=self.o.name or src.name, experimental=self.o.experimental)
        if self.o.force_route == "master":
            pl.steps = [s for s in pl.steps if s not in ("sidon", "level_parts")]
        elif self.o.force_route == "sidon" and "sidon" not in pl.steps:
            pl.steps.insert(0, "sidon")
        self.note("Route: " + pl.line("en"))
        for w in pl.warnings:
            self.note("Warning: " + w)
        dur = pp["file"]["decoded_duration"]
        self._disk_check(dur)
        ref = load_mono(self.o.reference, SR48) if self.o.reference else None
        probe = self._probe(src, pp, pl, ref)
        if probe:
            fails = [g for g in probe["gates"] if g["status"] == "fail" and g["name"] in ("quality", "not_worse")]
            if fails and "sidon" not in pl.steps:
                self.note("Probe failed on quality with the light route; switching to Sidon for the full run")
                pl.steps.insert(0, "sidon")
        target_len = stages.source_len48(src)
        rounds = []
        x = self._restore_track(src, pl, target_len=target_len)
        final, info = self._finish(x, pl, "full", reference=ref)
        final, mapping, exp_len, trim_info = self._trim(final, src, pp)
        windows = [(w["start"], w["end"]) for w in pp["windows"]]
        segs = [(mapping.out_t(p["start"]), mapping.out_t(p["end"])) for p in pp["level"]["parts"]] if len(pp["level"]["parts"]) > 1 else None
        v = None
        history = []
        for rnd in range(1, T.MAX_ROUNDS + 1):
            # the route can change in the fix loop (light route -> Sidon), so the seams too
            seams = [mapping.out_t(t) for t in stages.seams(dur, stages.last_chunk_s)] if "sidon" in pl.steps else []
            v = verify.verify(src, final, windows=windows, input_scores=pp["windows"], mapping=mapping,
                              reference=self.o.reference, expected_len=exp_len, target=self.o.target,
                              asr_model=self.o.asr_model, language=self.o.language, seams=seams, segments=segs)
            st = {g["name"]: g["status"] for g in v["gates"]}
            rounds.append({"round": rnd, "gates": st})
            self.note(f"Verify round {rnd}: " + ", ".join(f"{k} {s}" for k, s in st.items()))
            keep = self.work / f"round{rnd}.wav"
            shutil.copyfile(final, keep)
            history.append((_badness(st), rnd, keep, v, mapping))
            if v["ok"] or rnd == T.MAX_ROUNDS:
                break
            if rnd > 1 and history[-1][0] > history[-2][0]:
                self.note(f"Round {rnd} made things worse than round {rnd - 1}; stopping the fix loop")
                break
            fixed = self._fix(v, src, pp, pl, x, ref, final, mapping)
            if fixed is None:
                self.note("No fix applies to the failing gates; delivering as is with the failures reported")
                break
            x, final, mapping = fixed
        best = min(history, key=lambda h: (h[0], h[1]))
        if best[1] != history[-1][1]:
            self.note(f"Delivering round {best[1]}: later fixes did not improve the gates")
        _, _, final, v, mapping = best
        rounds.append({"delivered_round": best[1]})
        return self._deliver(src, pp, pl, final, v, rounds, info, trim_info, probe, mapping)

    # ------------------------------------------------------------------ fix loop
    def _fix(self, v: dict, src: Path, pp: dict, pl, x: np.ndarray, ref, final: Path, mapping):
        failing = {g["name"]: g for g in v["gates"] if g["status"] == "fail"}
        if "quality" in failing or "not_worse" in failing:
            if "sidon" not in pl.steps:
                why = " and ".join(n.replace("_", " ") for n in ("quality", "not_worse") if n in failing)
                self.note(f"Fix: the light route failed {why} → switching to Sidon")
                pl.steps.insert(0, "sidon")
                x = self._restore_track(src, pl)
                f, _ = self._finish(x, pl, "full", reference=ref)
                f, mapping, _, _ = self._trim(f, src, pp)
                return x, f, mapping
            nw = failing.get("not_worse", {}).get("details", {}).get("windows", [])
            clean_in = [r for r in nw if r["in"]["ovrl"] >= T.TARGET_OVRL and r["in"]["bak"] >= T.TARGET_BAK]
            if "quality" not in failing and clean_in and not getattr(self, "_tried_light", False):
                # the input already sounds like a close mic there: restoration cannot beat it, so try
                # the light route and keep whichever round passes more gates
                self._tried_light = True
                self.note("Fix: input already at the podcast-mic target and restoration did not beat it "
                          "→ trying the light route (mix only)")
                lpl = type(pl)(**{**pl.__dict__, "steps": [s for s in pl.steps if s not in ("sidon", "level_parts")]})
                x2 = self._restore_track(src, lpl)
                f, _ = self._finish(x2, lpl, "light", reference=ref)
                f, mapping, _, _ = self._trim(f, src, pp)
                return x2, f, mapping
        if "sync" in failing:
            det = failing["sync"]["details"]
            med = det.get("median_ms")
            if med is not None and det.get("reliable") and abs(med) > T.SYNC_MS:
                self.note(f"Fix: output is {med:+.2f} ms off → shifting it back")
                y = stages.read48(final)
                k = int(round(med * SR48 / 1000))
                y2 = np.concatenate([y[k:], np.zeros(k, np.float32)]) if k > 0 else np.concatenate([np.zeros(-k, np.float32), y[:k]])
                write_wav(final, np.stack([y2, y2], 1), SR48)
                return x, final, mapping
        if "loudness" in failing:
            self.note("Fix: loudness/peaks off → re-mixing with recomputed gain and limiter")
            pre = self.work / "full_premaster.wav"
            master_mod.master(pre, self.work / "full_master.wav", target=self.o.target, limit=0.8)
            f, mapping, _, _ = self._trim(self.work / "full_master.wav", src, pp)
            return x, f, mapping
        if "timbre" in failing and ref is not None:
            self.note("Fix: timbre off the reference → one more matching pass")
            pre = stages.read48(self.work / "full_premaster.wav")
            cs, d = match.correction(ref, stages.read48(self.work / "full_master.wav"), SR48)
            y = match.apply_fir(pre, match.design_fir(cs, d, SR48))
            stages.write48(self.work / "full_premaster.wav", y)
            master_mod.master(self.work / "full_premaster.wav", self.work / "full_master.wav", target=self.o.target)
            f, mapping, _, _ = self._trim(self.work / "full_master.wav", src, pp)
            return x, f, mapping
        if "words" in failing or "speaker" in failing:
            self.note("Fix: words or voice changed in a window → light route (no Sidon) on that stretch")
            bad = []
            for name in ("words", "speaker"):
                if name in failing:
                    for w in failing[name]["details"].get("windows", []):
                        if (name == "words" and not w["ok"]) or (name == "speaker" and w["similarity"] < T.SPEAKER_MIN):
                            bad.append((w["start"], w["end"]))
            if not bad or "sidon" not in pl.steps:
                return None
            light = stages.read48(stages.mono_source(src, pl.params.get("channel")))
            light = light[: len(x)] if len(light) >= len(x) else np.pad(light, (0, len(x) - len(light)))
            light, _ = match.match(x, light, SR48, passes=1)
            segs = vad.speech_segments(load_mono(src))
            y = x.copy()
            for a, b in bad:
                a2 = vad.nearest_pause(segs, len(x) / SR48, a, 5)
                b2 = vad.nearest_pause(segs, len(x) / SR48, b, 5)
                i, j = int(a2 * SR48), int(b2 * SR48)
                n = int(0.03 * SR48)
                ramp = np.linspace(0, 1, n, dtype=np.float32)
                y[i:j] = light[i:j]
                y[i:i + n] = x[i:i + n] * (1 - ramp) + light[i:i + n] * ramp
                y[j - n:j] = light[j - n:j] * (1 - ramp) + x[j - n:j] * ramp
                self.note(f"  light route on {a2:.1f}-{b2:.1f} s")
            f, _ = self._finish(y, type(pl)(**{**pl.__dict__, "steps": [s for s in pl.steps if s != "level_parts"]}), "full", reference=ref)
            f, mapping, _, _ = self._trim(f, src, pp)
            return y, f, mapping
        return None

    # ------------------------------------------------------------------ trimming
    def _trim(self, final: Path, src: Path, pp: dict):
        """Apply a requested trim. Returns (path, mapping, expected_len, what was cut)."""
        n = stages.read48(final)
        total = len(n)
        a, b = 0.0, total / SR48
        cut = []
        if self.o.trim_start is not None:
            a = float(self.o.trim_start)
        if self.o.trim_end is not None:
            b = float(self.o.trim_end)
        if self.o.trim_tail:
            segs = vad.speech_segments(load_mono(src))
            if segs:
                last = segs[-1][1]
                b = min(b, last + 2.0)
        if a <= 0 and b >= total / SR48 - 1e-6:
            return final, verify.Mapping(), total, []
        x16 = load_mono(src)
        for lo, hi, where in ((0.0, a, "start"), (b, total / SR48, "end")):
            if hi - lo <= 0.001:
                continue
            seg = x16[int(lo * 16000): int(hi * 16000)]
            text = ""
            if asr.available(self.o.asr_model) and len(seg) > 8000:
                text, _, _ = asr.transcribe(seg, model=self.o.asr_model, language=self.o.language)
            sp = vad.speech_ratio(vad.speech_segments(seg), len(seg) / 16000) if len(seg) > 8000 else 0.0
            cut.append({"where": where, "from_s": round(lo, 3), "to_s": round(hi, 3), "duration_s": round(hi - lo, 3),
                        "speech_ratio": round(sp, 2), "recognised_text": text})
        y = n[int(round(a * SR48)): int(round(b * SR48))]
        from .audio import fade
        y = fade(y, SR48, fade_in=0.02 if a > 0 else 0.0, fade_out=0.5 if b < total / SR48 else 0.0)
        out = final.with_name(final.stem + "_trim.wav")
        write_wav(out, np.stack([y, y], 1), SR48)
        return out, verify.Mapping(offset=-a), len(y), cut

    # ------------------------------------------------------------------ multi-source session
    def _process_session(self) -> dict:
        timeline = Path(self.o.sync_to) if self.o.sync_to else self.inputs[0]
        self.note(f"Session: {len(self.inputs)} source(s) on the timeline of {timeline.name}")
        tl16 = load_mono(timeline)
        tl_len = stages.source_len48(timeline)
        sources = []
        for p in self.inputs:
            pp = analyze_mod.analyze(p)
            if not pp["file"].get("readable"):
                self.note(f"{p.name}: unreadable, skipped")
                continue
            x16 = load_mono(p)
            same, off = align.same_session(tl16, x16)
            if not same and p != timeline:
                self.note(f"{p.name}: envelope match {off.score:.2f} — not this session, skipped")
                continue
            dr = align.measure_drift(tl16, x16, guess=off.seconds) if p != timeline else align.Drift(0.0, 0.0, 0.0, [])
            s = analyze_mod.summary(pp)
            sources.append({"path": p, "passport": pp, "drift": dr, "score": off.score if p != timeline else 1.0, "q": s})
            self.note(f"{p.name}: match {off.score:.2f}, offset {dr.offset:+.3f} s, drift {dr.ppm:+.2f} ppm, "
                      f"OVRL {s.get('ovrl', 0):.2f}, SRMR {s.get('srmr', 0):.1f}")
        if not sources:
            raise RuntimeError("no usable source in this session")
        sources.sort(key=lambda s: (s["q"].get("ovrl", 0), s["q"].get("srmr", 0)), reverse=True)
        best = sources[0]
        self.note(f"Best source: {best['path'].name}")
        ref = load_mono(self.o.reference, SR48) if self.o.reference else None
        tracks = []
        for s in sources:
            pl = make_plan(s["passport"], name=s["path"].name, glue_with_sidon=True, experimental=self.o.experimental)
            self.note("Route: " + pl.line("en"))
            x = self._restore_track(s["path"], pl)
            x = align.stretch(x, SR48, s["drift"].ppm) if abs(s["drift"].ppm) > 0.05 else x
            x = align.place(x, SR48, s["drift"].offset, tl_len)
            cov = np.zeros(tl_len, dtype=bool)
            a = int(max(0, round(s["drift"].offset * SR48)))
            b = int(min(tl_len, a + stages.source_len48(s["path"]) * (1 + s["drift"].ppm * 1e-6)))
            cov[a:b] = True
            tracks.append({"src": s, "plan": pl, "x": x, "cov": cov})
            if cov.all():
                break
        # assemble: best everywhere it has signal, others fill gaps, switches in pauses
        main = tracks[0]
        y = main["x"].copy()
        seams_tl = []
        for t in tracks[1:]:
            gap = (~main["cov"]) & t["cov"]
            if not gap.any():
                continue
            t["x"], _ = match.match(main["x"][main["cov"]], t["x"], SR48, passes=2)
            lm, lf = _speech_lufs(main["x"][main["cov"]]), _speech_lufs(t["x"][gap])
            # a filler that is silent there has no loudness to match: leave its level alone
            gl = 10 ** (float(np.clip(lm - lf, -20.0, 20.0)) / 20) if np.isfinite(lm) and np.isfinite(lf) else 1.0
            fill = t["x"] * gl
            edges = np.flatnonzero(np.diff(np.concatenate([[0], gap.astype(np.int8), [0]])))
            n = int(0.03 * SR48)
            for i, j in zip(edges[::2], edges[1::2]):
                y[i:j] = fill[i:j]
                for e in (i, j):
                    lo, hi = max(0, e - n // 2), min(len(y), e + n // 2)
                    r = np.linspace(0, 1, hi - lo, dtype=np.float32)
                    if e == i:
                        y[lo:hi] = main["x"][lo:hi] * (1 - r) + fill[lo:hi] * r
                    else:
                        y[lo:hi] = fill[lo:hi] * (1 - r) + main["x"][lo:hi] * r
                    seams_tl.append(e / SR48)
            self.note(f"{t['src']['path'].name} fills {gap.sum() / SR48:.1f} s where {main['src']['path'].name} has no signal")
        for t in tracks:  # an hour at 48 kHz is ~0.7 GB per copy: keep only the assembled track
            t["x"] = t["cov"] = None
        del tl16
        pl = main["plan"]
        mp = verify.Mapping(offset=main["src"]["drift"].offset, ppm=main["src"]["drift"].ppm)
        if pl.params.get("parts"):
            # parts were measured on the source's own clock: move them onto the session timeline
            tl_dur = tl_len / SR48
            parts = [{**p, "start": max(0.0, mp.out_t(p["start"])), "end": min(tl_dur, mp.out_t(p["end"]))}
                     for p in pl.params["parts"]]
            parts = [p for p in parts if p["end"] - p["start"] > 1.0]
            pl = type(pl)(**{**pl.__dict__, "params": {**pl.params, "parts": parts}})
            if len(parts) < 2:
                pl.steps = [s for s in pl.steps if s != "level_parts"]
        final, info = self._finish(y, pl, "full", reference=ref)
        final, mp_trim, exp_len, trim_info = self._trim(final, timeline, main["src"]["passport"])
        mp = verify.Mapping(offset=mp.offset + mp_trim.offset, ppm=mp.ppm)
        pp = main["src"]["passport"]
        windows = [(w["start"], w["end"]) for w in pp["windows"]]
        seams = [mp.out_t(t) for t in stages.seams(pp["file"]["decoded_duration"], stages.last_chunk_s)] + [t + mp_trim.offset for t in seams_tl]
        v = verify.verify(main["src"]["path"], final, windows=windows, input_scores=pp["windows"], mapping=mp,
                          reference=self.o.reference, expected_len=exp_len, target=self.o.target,
                          asr_model=self.o.asr_model, language=self.o.language, seams=seams)
        rounds = [{"round": 1, "gates": {g["name"]: g["status"] for g in v["gates"]}}]
        self.note("Verify: " + ", ".join(f"{k} {s}" for k, s in rounds[0]["gates"].items()))
        info["session"] = [{"path": str(s["src"]["path"]), "offset_s": s["src"]["drift"].offset, "ppm": s["src"]["drift"].ppm,
                            "match": s["src"]["score"], "quality": s["src"]["q"]} for s in tracks]
        return self._deliver(main["src"]["path"], pp, pl, final, v, rounds, info, trim_info, None, mp,
                             timeline=timeline, extra_seams=seams_tl)

    # ------------------------------------------------------------------ outputs
    def _deliver(self, src: Path, pp: dict, pl, final: Path, v: dict, rounds, info, trim_info, probe, mapping,
                 timeline: Path | None = None, extra_seams=None) -> dict:
        stem = self.o.name or src.stem
        out_wav = self.out / f"{stem}.clean.wav"
        shutil.copyfile(final, out_wav)
        files = report.listening_probes(src, out_wav, self.out, pp, mapping)
        if extra_seams:
            files["junction_probe"] = str(report.junction_probe(out_wav, extra_seams, self.out / "junction_probe.wav"))
        metrics_doc = {
            "input": str(src), "output": str(out_wav), "timeline": str(timeline or src),
            "passport": pp, "plan": pl.to_dict(), "verify": v, "rounds": rounds, "stages": info,
            "trimmed": trim_info, "probe": probe, "events": self.events, "files": files,
            "seconds": round(time.time() - self.t0, 1),
        }
        (self.out / "metrics.json").write_text(json.dumps(metrics_doc, indent=2, ensure_ascii=False, default=str))
        (self.out / "report.md").write_text(report.render(metrics_doc, lang=self.o.lang))
        if not self.o.keep_work:
            shutil.rmtree(self.work, ignore_errors=True)
        self.note(f"Done: {out_wav}")
        return metrics_doc

    def _refuse(self, src: Path, pp: dict) -> dict:
        doc = {"input": str(src), "passport": pp, "refused": True,
               "reason": "no decodable audio stream; ffmpeg errors: " + "; ".join(pp["file"].get("probe_errors", [])[:3])}
        (self.out / "metrics.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False))
        (self.out / "report.md").write_text(report.render_refusal(doc, lang=self.o.lang))
        return doc


def _badness(st: dict) -> int:
    """Rank a verify round: quality failures weigh most (a fix must never bring the echo back)."""
    w = {"quality": 10, "not_worse": 10, "words": 3, "speaker": 3, "sync": 3, "loudness": 2, "timbre": 1, "artifacts": 2}
    return sum(w.get(k, 1) for k, v in st.items() if v == "fail")


def _speech_lufs(x48: np.ndarray) -> float:
    from . import loudness

    return loudness.integrated(x48, SR48)


def process(inputs: list[Path], opts: Options) -> dict:
    return Run(inputs, opts).process()
