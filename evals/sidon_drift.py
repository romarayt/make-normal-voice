"""One hour through the real Sidon: offset to the input at 20 points must stay under 1 ms.

Also runs the reference demo's gluing on the same hour, to show the 5.6 ppm drift it had.
    VR_EVAL_DATA=... python evals/sidon_drift.py [minutes]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "skills" / "voice-rescue" / "src"))

from fetch_data import data_dir  # noqa: E402
from voice_rescue import align  # noqa: E402
from voice_rescue.engines import sidon  # noqa: E402


def build_long(minutes: float) -> np.ndarray:
    meta = json.loads((data_dir() / "clean_en" / "meta.json").read_text())
    parts, total = [], 0
    rng = np.random.default_rng(0)
    while total < minutes * 60 * 16000:
        m = meta[int(rng.integers(len(meta)))]
        x, sr = sf.read(str(data_dir() / "clean_en" / f"{m['id']}.wav"), dtype="float32")
        x = soxr.resample(x, sr, 16000).astype(np.float32)
        parts += [x, np.zeros(int(rng.uniform(0.3, 1.2) * 16000), np.float32)]
        total += len(x) + len(parts[-1])
    y = np.concatenate(parts)[: int(minutes * 60 * 16000)]
    return (0.9 * y / np.abs(y).max()).astype(np.float32)


def naive(eng: sidon.Sidon, x16: np.ndarray) -> np.ndarray:
    """The reference demo recipe: 96 s chunks, cache frame, [:-960], glue."""
    import torch

    wav = torch.nn.functional.pad(torch.from_numpy(x16), (0, 24000))
    outs, cache = [], None
    with torch.inference_mode():
        for ch in wav.split(16000 * 96):
            f = eng.pre(torch.nn.functional.pad(ch, (160, 160)).numpy(), sampling_rate=16000, return_tensors="pt")["input_features"]
            f = eng.fe(f)["last_hidden_state"]
            if cache is not None:
                f = torch.cat([cache, f], dim=1)
            outs.append(eng.dec(f.transpose(1, 2)).view(-1)[:-960].float().cpu())
            cache = f[:, -1:]
    y = torch.cat(outs).numpy()[: len(x16) * 3]
    return np.pad(y, (0, len(x16) * 3 - len(y)))


def offsets(x16: np.ndarray, y48: np.ndarray, n: int = 20) -> list[float]:
    y16 = soxr.resample(y48, 48000, 16000, quality="VHQ").astype(np.float32)
    out = []
    for c in np.linspace(20, len(x16) / 16000 - 20, n):
        o = align.local_offset(x16, y16, 16000, c, win=20.0, search=0.06, guess=0.0)
        out.append(round(-o.seconds * 1000, 3))  # positive = output late
    return out


def main():
    minutes = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    x16 = build_long(minutes)
    eng = sidon.Sidon("cpu")
    tmp = data_dir() / "drift_out.wav"
    t = time.time()
    eng.restore(x16, tmp, target=len(x16) * 3, progress=lambda m: print(m, flush=True))
    rt = time.time() - t
    y, _ = sf.read(str(tmp), dtype="float32")
    ours = offsets(x16, y)
    res = {"minutes": minutes, "seconds": round(rt, 1), "x_realtime": round(minutes * 60 / rt, 2),
           "length_exact": len(y) == len(x16) * 3, "ours_ms": ours}
    print(json.dumps(res), flush=True)
    if "--naive" in sys.argv:  # the reference gluing decodes 96 s at once: ~20 GB on CPU, off by default
        t = time.time()
        yn = naive(eng, x16)
        res["naive_ms"] = offsets(x16, yn)
        res["naive_seconds"] = round(time.time() - t, 1)
    k = np.polyfit(np.linspace(20, minutes * 60 - 20, len(ours)), ours, 1)
    res["ours_drift_ppm"] = round(k[0] * 1000, 3)
    if "naive_ms" in res:
        kn = np.polyfit(np.linspace(20, minutes * 60 - 20, len(ours)), res["naive_ms"], 1)
        res["naive_drift_ppm"] = round(kn[0] * 1000, 3)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "sidon_drift.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    tmp.unlink()


if __name__ == "__main__":
    main()
