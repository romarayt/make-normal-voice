"""Derive the gate thresholds from data instead of guessing them.

* podcast-mic target: median DNSMOS OVRL / BAK and SRMR of the clean references, rounded down;
* speaker threshold: between "same person, restored" and "two different people";
* word margin: what a correct restoration costs in confident-word loss beyond the recogniser noise.

    VR_EVAL_DATA=... python evals/derive_thresholds.py   ->  evals/results/thresholds.json
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "skills" / "make-normal-voice" / "src"))

from bench import work  # noqa: E402


def main():
    from make_normal_voice import asr, speaker
    from make_normal_voice.audio import load_mono

    clean = json.loads((work() / "clean_stats.json").read_text())
    scores = json.loads((work() / "scores.json").read_text())
    res = {}
    for lang in ("en", "ru", "all"):
        c = [s for s in clean if lang == "all" or s["lang"] == lang]
        res[f"clean_{lang}"] = {k: {"median": round(float(np.median([s[k] for s in c])), 3),
                                    "p25": round(float(np.percentile([s[k] for s in c], 25)), 3)}
                                for k in ("ovrl", "bak", "sig", "srmr")} | {"n": len(c)}
    # different people: pairs of clean references from different speakers
    refs = {}
    for s in clean:
        refs.setdefault(s["speaker"], s)
    emb = {}
    for spk, s in refs.items():
        emb[spk] = speaker.embedding(load_mono(s["path"]))
    diff = [float(np.dot(emb[a], emb[b])) for a, b in itertools.combinations(emb, 2)]
    same_restored = [v["speaker"] for v in scores.values() if v.get("candidate") == "sidon" and v.get("speaker") is not None]
    res["speaker"] = {"different_people_max": round(max(diff), 3), "different_people_p99": round(float(np.percentile(diff, 99)), 3),
                      "same_person_restored_min": round(min(same_restored), 3),
                      "same_person_restored_p1": round(float(np.percentile(same_restored, 1)), 3)}
    # word margin: confident-word loss of Sidon outputs vs their (degraded) inputs, minus recogniser noise
    rows = json.loads((work() / "set" / "set.json").read_text())["rows"]
    margins = []
    for r in rows:
        if r["class"] in ("music",):
            continue
        o = work() / "out" / "sidon" / Path(r["input"]).name
        if not o.exists():
            continue
        x, sr = sf.read(r["input"], dtype="float32")
        y, sr2 = sf.read(str(o), dtype="float32")
        x16 = soxr.resample(x, sr, 16000).astype(np.float32)
        y16 = soxr.resample(y, sr2, 16000).astype(np.float32)
        _, wa, lang = asr.transcribe(x16, language=r["lang"], words=True)
        tb, _, _ = asr.transcribe(x16, language=r["lang"], beam_size=1)
        to, _, _ = asr.transcribe(y16, language=r["lang"])
        floor, n, _ = asr.confident_loss(wa, tb)
        loss, _, _ = asr.confident_loss(wa, to)
        if n >= 5:
            margins.append({"class": r["class"], "lang": r["lang"], "excess": round(loss - floor, 3), "n": n})
    ex = [m["excess"] for m in margins]
    res["words"] = {"excess_p50": round(float(np.median(ex)), 3), "excess_p90": round(float(np.percentile(ex, 90)), 3),
                    "excess_p95": round(float(np.percentile(ex, 95)), 3), "excess_max": round(max(ex), 3),
                    "n": len(ex), "worst": sorted(margins, key=lambda m: -m["excess"])[:8]}
    out = HERE / "results" / "thresholds.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
