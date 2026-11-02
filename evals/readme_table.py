"""Render the README benchmark tables from evals/results/bench.json (no hand-typed numbers)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABELS = {
    "en": {"clean": "clean (repeatability)", "far_echo": "far mic, room echo", "far_echo_noise": "far mic, echo + room noise",
           "laptop": "laptop-like (small room, band-limited, AAC)", "noise_dry": "dry voice, loud noise (5 dB SNR)",
           "clipping": "clipping", "phone": "phone (300-3400 Hz, 8 kHz µ-law)", "codec_lowbit": "MP3 24 kbps",
           "hum": "50/60 Hz hum", "quiet": "very quiet (-40 LUFS, 16-bit)", "dropouts": "digital dropouts",
           "music": "music under the voice"},
    "ru": {"clean": "чистая (повторяемость)", "far_echo": "дальний микрофон, эхо", "far_echo_noise": "дальний, эхо + шум комнаты",
           "laptop": "как ноутбук (малая комната, полоса, AAC)", "noise_dry": "сухой голос, сильный шум (5 дБ)",
           "clipping": "клиппинг", "phone": "телефон (300-3400 Гц, 8 кГц µ-law)", "codec_lowbit": "MP3 24 кбит/с",
           "hum": "фон сети 50/60 Гц", "quiet": "очень тихо (-40 LUFS, 16 бит)", "dropouts": "цифровые обрывы",
           "music": "музыка под голосом"},
}
ROUTE = {"clean": "mix only", "default": "Sidon → mix"}
OTHERS = ["deepfilternet", "clearvoice", "voicefixer", "resemble-enhance", "sidon-onnx-int8", "sidon-onnx-fp16",
          "sidon-declip", "sidon-notch"]


def fmt(v, d=2, lang="en"):
    if v is None:
        return "–"
    s = f"{v:.{d}f}"
    return s.replace(".", ",") if lang == "ru" else s


def main(lang: str = "en"):
    b = json.loads((HERE / "results" / "bench.json").read_text())["by_class"]
    order = list(LABELS["en"])
    if lang == "ru":
        head = ("| Класс | Вход OVRL | Sidon OVRL / BAK | Sidon PESQ / ESTOI | Sidon WER / голос | Лучший другой по OVRL (его WER) |\n"
                "|---|---|---|---|---|---|")
    else:
        head = ("| Class | Input OVRL | Sidon OVRL / BAK | Sidon PESQ / ESTOI | Sidon WER / voice | Best other by OVRL (its WER) |\n"
                "|---|---|---|---|---|---|")
    lines = [head]
    for cls in order:
        c = b.get(cls)
        if not c or "sidon" not in c:
            continue
        s, i = c["sidon"], c["input"]
        others = [(n, c[n]) for n in OTHERS if n in c and c[n].get("ovrl") is not None and not n.startswith("sidon-onnx")]
        best = max(others, key=lambda kv: kv[1]["ovrl"]) if others else None
        bo = (f"{best[0]} {fmt(best[1]['ovrl'], lang=lang)} (WER {fmt(best[1]['wer'], lang=lang)})" if best else "–")
        lines.append(f"| {LABELS[lang][cls]} | {fmt(i['ovrl'], lang=lang)} | {fmt(s['ovrl'], lang=lang)} / {fmt(s['bak'], lang=lang)} | "
                     f"{fmt(s['pesq'], lang=lang)} / {fmt(s['estoi'], lang=lang)} | {fmt(s['wer'], lang=lang)} / {fmt(s['speaker'], lang=lang)} | {bo} |")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "en")
