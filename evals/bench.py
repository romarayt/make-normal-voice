"""Benchmark: restoration candidates × track classes, on open speech (EN + RU).

    VR_EVAL_DATA=... python evals/bench.py prepare      # pick references, degrade them
    VR_EVAL_DATA=... python evals/bench.py run <cand>   # one candidate at a time (heavy ones use all cores)
    VR_EVAL_DATA=... python evals/bench.py score        # metrics for every output
    VR_EVAL_DATA=... python evals/bench.py report       # tables -> evals/results/

Candidates run in their own environments (see evals/README.md); `ours:*` run in voice-rescue's.
Outputs are scored as they come out of each engine (no mixing), aligned to the clean reference.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from scipy import signal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "skills" / "voice-rescue" / "src"))

from degrade import CLASSES, build  # noqa: E402
from fetch_data import data_dir  # noqa: E402

CAND_ENVS = Path(os.environ.get("VR_CANDIDATES", data_dir().parent / "cand"))
CANDIDATES = {
    "master-only": ("ours", "master"),
    "sidon": ("ours", "sidon"),
    "sidon-onnx-int8": ("ours", "sidon-onnx-int8"),
    "sidon-onnx-fp16": ("ours", "sidon-onnx-fp16"),
    "sidon-declip": ("ours", "sidon-declip"),     # hypothesis: adeclip before restoration (clipping class)
    "sidon-notch": ("ours", "sidon-notch"),       # hypothesis: mains-hum notch before restoration (hum class)
    "deepfilternet": ("dfn", "run_deepfilternet"),
    "clearvoice": ("clearvoice", "run_clearvoice"),
    "voicefixer": ("voicefixer", "run_voicefixer"),
    "resemble-enhance": ("resemble", "run_resemble"),
}
N_EN, N_RU = 6, 6


def work() -> Path:
    d = data_dir() / "bench"
    d.mkdir(exist_ok=True)
    return d


def prepare():
    from voice_rescue import metrics
    from voice_rescue.audio import load_mono

    data = data_dir()
    en = json.loads((data / "clean_en" / "meta.json").read_text())
    ru = json.loads((data / "clean_ru" / "meta.json").read_text())
    # clean-reference statistics for the target: every clean utterance we have
    stats = []
    for m, sub in [(m, "clean_en") for m in en] + [(m, "clean_ru") for m in ru]:
        p = data / sub / f"{m['id']}.wav"
        s = metrics.score(load_mono(p))
        stats.append({**m, "path": str(p), **s})
        print(f"clean {m['id']}: {s}", flush=True)
    (work() / "clean_stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    # references: EN one per speaker; RU the cleanest FLEURS utterances (crowd-read, so filter by OVRL)
    by_spk = {}
    for s in stats:
        if s["lang"] == "en" and s["speaker"] not in by_spk:
            by_spk[s["speaker"]] = s
    refs = list(by_spk.values())[:N_EN]
    # RU: one per speaker, the cleanest first (home recordings vary)
    seen = set()
    for s in sorted([s for s in stats if s["lang"] == "ru"], key=lambda s: -s["ovrl"]):
        if s["speaker"] not in seen and len(seen) < N_RU:
            seen.add(s["speaker"])
            refs.append(s)
    rows = build(data, work() / "set", refs)
    print(f"{len(rows)} degraded inputs from {len(refs)} references", flush=True)


def run(name: str):
    env, runner = CANDIDATES[name]
    rows = json.loads((work() / "set" / "set.json").read_text())["rows"]
    only = {"sidon-declip": "clipping", "sidon-notch": "hum"}.get(name)
    inputs = [r["input"] for r in rows if only is None or r["class"] == only]
    lst = work() / f"inputs_{name}.json"
    lst.write_text(json.dumps(inputs))
    out = work() / "out" / name
    if env == "ours":
        py = sys.executable
        args = [py, str(HERE / "candidates" / "ours.py"), str(lst), str(out), runner]
    else:
        py = str(CAND_ENVS / env / "venv" / "bin" / "python")
        args = [py, str(HERE / "candidates" / f"{runner}.py"), str(lst), str(out)]
    t = time.time()
    subprocess.run(args, check=False, cwd=str(CAND_ENVS / env) if env != "ours" else None,
                   env={**os.environ, "VR_EVAL_DATA": str(data_dir())})
    print(f"{name}: {time.time() - t:.0f} s", flush=True)


def _align(ref: np.ndarray, y: np.ndarray, sr: int, search: float = 0.15) -> tuple[np.ndarray, float]:
    n = min(len(ref), len(y))
    m = int(search * sr)
    c = signal.correlate(y[:n], ref[:n], mode="full", method="fft")
    mid = n - 1
    k = int(np.argmax(np.abs(c[mid - m: mid + m + 1]))) - m
    y2 = y[k:] if k > 0 else np.concatenate([np.zeros(-k, np.float32), y])
    y2 = y2[: len(ref)]
    return np.pad(y2, (0, len(ref) - len(y2))).astype(np.float32), k / sr


def score():
    from pesq import pesq
    from pystoi import stoi
    from voice_rescue import asr, metrics, speaker

    rows = json.loads((work() / "set" / "set.json").read_text())["rows"]
    res_p = work() / "scores.json"
    res = json.loads(res_p.read_text()) if res_p.exists() else {}
    names = ["input"] + [c for c in CANDIDATES if (work() / "out" / c).exists()]
    ref_text_p = work() / "ref_text.json"
    ref_text = json.loads(ref_text_p.read_text()) if ref_text_p.exists() else {}
    for r in rows:  # words are judged against what the recogniser hears in the clean original
        if r["id"] not in ref_text:
            ref, rsr = sf.read(r["ref"], dtype="float32")
            ref_text[r["id"]] = asr.transcribe(soxr.resample(ref, rsr, 16000).astype(np.float32), language=r["lang"])[0]
    ref_text_p.write_text(json.dumps(ref_text, ensure_ascii=False, indent=0))
    for name in names:
        timing = {}
        if name != "input":
            tp = work() / "out" / name / "timing.json"
            timing = json.loads(tp.read_text()) if tp.exists() else {}
        for r in rows:
            key = f"{name}|{Path(r['input']).name}"
            if key in res and not res[key].get("failed"):
                continue
            p = Path(r["input"]) if name == "input" else work() / "out" / name / Path(r["input"]).name
            if not p.exists():
                res[key] = {"failed": True}
                continue
            y, sr = sf.read(str(p), dtype="float32", always_2d=True)
            y16 = soxr.resample(y[:, 0], sr, 16000, quality="VHQ").astype(np.float32)
            ref, rsr = sf.read(r["ref"], dtype="float32")
            ref16 = soxr.resample(ref, rsr, 16000, quality="VHQ").astype(np.float32)
            y16a, lag = _align(ref16, y16, 16000)
            s = metrics.score(y16a)
            try:
                s["pesq"] = float(pesq(16000, ref16, y16a, "wb"))
            except Exception:  # noqa: BLE001 - pesq refuses some silent/garbage outputs
                s["pesq"] = None
            s["estoi"] = float(stoi(ref16, y16a, 16000, extended=True))
            hyp, _, _ = asr.transcribe(y16a, language=r["lang"])
            s["wer"] = asr.wer(ref_text[r["id"]], hyp)
            s["speaker"] = speaker.similarity(ref16, y16a)
            s["lag_ms"] = round(lag * 1000, 2)
            dur = len(ref16) / 16000
            t = timing.get(p.name)
            s["rtf"] = (t / dur) if t else None
            s.update({"class": r["class"], "lang": r["lang"], "ref_id": r["id"], "candidate": name})
            res[key] = s
            print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}, flush=True)
            res_p.write_text(json.dumps(res, indent=0))
    res_p.write_text(json.dumps(res, indent=0))


def report():
    res = json.loads((work() / "scores.json").read_text())
    clean = json.loads((work() / "clean_stats.json").read_text())
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    keys = ["ovrl", "sig", "bak", "srmr", "pesq", "estoi", "wer", "speaker", "rtf"]
    table = {}
    for v in res.values():
        if v.get("failed"):
            continue
        table.setdefault(v["class"], {}).setdefault(v["candidate"], []).append(v)
    agg = {}
    for cls, cands in table.items():
        for c, vs in cands.items():
            agg.setdefault(cls, {})[c] = {
                k: (float(np.median([x[k] for x in vs if x.get(k) is not None])) if any(x.get(k) is not None for x in vs) else None)
                for k in keys} | {"n": len(vs)}
    cl = {k: float(np.median([s[k] for s in clean])) for k in ("ovrl", "sig", "bak", "srmr")}
    cl_en = {k: float(np.median([s[k] for s in clean if s["lang"] == "en"])) for k in ("ovrl", "sig", "bak", "srmr")}
    (out / "bench.json").write_text(json.dumps({"by_class": agg, "clean_median": cl, "clean_median_en": cl_en,
                                                "n_clean": len(clean)}, indent=1))
    lines = ["| Class | Candidate | OVRL | BAK | SRMR | PESQ | ESTOI | WER | Speaker | RTF |", "|---|---|---|---|---|---|---|---|---|---|"]
    order = ["input"] + list(CANDIDATES)
    for cls in CLASSES:
        for c in order:
            a = agg.get(cls, {}).get(c)
            if not a:
                continue
            def f(k, d=2, a=a):
                return "–" if a[k] is None else f"{a[k]:.{d}f}"
            lines.append(f"| {cls} | {c} | {f('ovrl')} | {f('bak')} | {f('srmr', 1)} | {f('pesq')} | {f('estoi')} | "
                         f"{f('wer')} | {f('speaker')} | {f('rtf')} |")
    (out / "bench.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("clean median", cl)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prepare":
        prepare()
    elif cmd == "run":
        for n in sys.argv[2:]:
            run(n)
    elif cmd == "score":
        score()
    elif cmd == "report":
        report()
