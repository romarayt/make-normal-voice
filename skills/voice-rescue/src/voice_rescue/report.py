"""Human-facing outputs: report.md, before/after listening probes, junction probe, spectrograms.

The agent cannot hear. The report says "by measurement" and hands over short probes at equal
loudness so a person can decide by ear; the ear outranks the numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import ffmpeg, loudness, vad
from .audio import dual_mono, load_channel, load_mono, write_wav

SR48 = 48000


def loudness_matched_copy(src: Path, dst: Path, *, start: float, duration: float, like: Path | None = None,
                          target: float = -16.0) -> Path:
    """A stretch of the source as 48 kHz stereo at the same loudness as `like` (or `target`)."""
    x = load_mono(src, SR48, start=start, duration=duration)
    want = ffmpeg.loudness(like).integrated if like else target
    have = loudness.integrated(x, SR48, channels=2)  # x is written as dual mono below
    g = 10 ** ((want - have) / 20) if np.isfinite(have) else 1.0
    y = x * g
    pk = np.max(np.abs(y)) if len(y) else 0
    if pk > 0.89:  # keep the louder raw input from clipping: lower it, never distort it
        y *= 0.89 / pk
    write_wav(dst, dual_mono(y), SR48)
    return dst


def spectrogram_png(x: np.ndarray, sr: int, path: Path, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=110)
    # a constant offset leaves digital silence at exactly zero power off DC (log10(0)); a fixed dither far
    # below the colour scale does not
    dither = 1e-8 * np.random.default_rng(0).standard_normal(len(x))
    ax.specgram(x + dither, NFFT=1024, Fs=sr, noverlap=768, cmap="magma", vmin=-130, vmax=-20)
    ax.set_ylim(0, min(sr / 2, 20000))
    ax.set_xlabel("s")
    ax.set_ylabel("Hz")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def listening_probes(src: Path, out_wav: Path, out_dir: Path, pp: dict, mapping, length: float = 30.0) -> dict:
    """Same 30 s, before and after, equal loudness; spectrograms of both."""
    wins = pp.get("windows") or []
    if wins:
        w = min(wins, key=lambda w: w["ovrl"])
        a_in = w["start"] + max(0.0, (w["end"] - w["start"] - length) / 2)
    else:
        a_in = 0.0
    a_out = mapping.out_t(a_in)
    if a_out < 0:
        a_in, a_out = a_in - a_out, 0.0
    after = out_dir / "ab_after_30s.wav"
    before = out_dir / "ab_before_30s.wav"
    y = load_channel(out_wav, SR48)[int(a_out * SR48): int((a_out + length) * SR48)]
    write_wav(after, dual_mono(y), SR48)
    loudness_matched_copy(src, before, start=a_in, duration=length, like=after)
    b = load_mono(before, SR48)
    files = {"ab_before": str(before), "ab_after": str(after), "ab_source_start_s": round(a_in, 2)}
    try:
        files["spectrogram_before"] = str(spectrogram_png(b, SR48, out_dir / "spectrogram_before.png", "before"))
        files["spectrogram_after"] = str(spectrogram_png(y, SR48, out_dir / "spectrogram_after.png", "after"))
    except Exception as e:  # noqa: BLE001 - pictures are a nicety, never a reason to fail
        files["spectrogram_error"] = str(e)
    return files


def junction_probe(track: Path, seams: list[float], dst: Path, each: float = 12.0) -> Path:
    """Short pieces around every seam, back to back, joined in pauses with 30 ms fades."""
    x = load_channel(track, SR48)
    x16 = load_mono(track, 16000)
    segs = vad.speech_segments(x16)
    dur = len(x) / SR48
    pieces = []
    n = int(0.03 * SR48)
    r = np.linspace(0, 1, n, dtype=np.float32)
    for s in seams[:12]:
        a = vad.nearest_pause(segs, dur, max(0.0, s - each / 2), 2.0)
        b = vad.nearest_pause(segs, dur, min(dur, s + each / 2), 2.0)
        p = x[int(a * SR48): int(b * SR48)].copy()
        if len(p) > 2 * n:
            p[:n] *= r
            p[-n:] *= r[::-1]
            pieces.append(p)
    y = np.concatenate(pieces) if pieces else np.zeros(SR48, np.float32)
    write_wav(dst, dual_mono(y), SR48)
    return dst


# ---------------------------------------------------------------- report.md

_T = {
    "en": {
        "title": "Voice track report", "was": "What it was", "done": "What was done", "check": "Checks",
        "cut": "What was cut", "nocut": "Nothing was cut: the track has the same length and start as the source.",
        "files": "Files", "unverified": "Not verified", "ear": "Measured, not listened to. The numbers are a "
        "proxy; listen to the before/after probes and trust your ear over them.",
        "gate": "Gate", "status": "Status", "result": "Result", "window": "Window", "before": "before", "after": "after",
        "route": "Route", "warn": "Warnings", "refused": "Could not process",
    },
    "ru": {
        "title": "Отчёт по дорожке", "was": "Что было", "done": "Что сделано", "check": "Проверка",
        "cut": "Что отрезано", "nocut": "Ничего не отрезано: дорожка той же длины и с того же момента, что исходник.",
        "files": "Файлы", "unverified": "Что не проверено", "ear": "Это замеры, а не прослушивание. Числа — "
        "лишь приближение: послушай пробы «до/после», ухо главнее замеров.",
        "gate": "Гейт", "status": "Итог", "result": "Результат", "window": "Окно", "before": "до", "after": "после",
        "route": "Маршрут", "warn": "Предупреждения", "refused": "Обработать не удалось",
    },
}
_STATUS = {"en": {"pass": "pass", "fail": "FAIL", "unverified": "not verified", "skipped": "skipped"},
           "ru": {"pass": "пройден", "fail": "НЕ ПРОЙДЕН", "unverified": "не проверен", "skipped": "пропущен"}}
_GATE = {"en": {"quality": "podcast-mic target", "not_worse": "not worse than input", "words": "words unchanged",
                "speaker": "same voice", "sync": "sync and length", "loudness": "loudness and peaks",
                "timbre": "timbre vs reference", "artifacts": "clicks, seams, lost speech"},
         "ru": {"quality": "цель «подкаст-микрофон»", "not_worse": "не хуже входа", "words": "слова не исказились",
                "speaker": "голос тот же", "sync": "синхрон и длина", "loudness": "громкость и пики",
                "timbre": "тембр к эталону", "artifacts": "щелчки, стыки, пропавшая речь"}}


def _mmss(t: float) -> str:
    t = max(0.0, float(t))
    return f"{int(t // 60)}:{t % 60:04.1f}"


def render(doc: dict, lang: str = "en") -> str:
    t = _T.get(lang, _T["en"])
    pp, pl, v = doc["passport"], doc["plan"], doc["verify"]
    fi = pp["file"]
    L = [f"# {t['title']}: {Path(doc['input']).name}", ""]
    L += [f"## {t['was']}", ""]
    L.append(f"- {fi.get('codec')} {fi.get('sample_rate')} Hz, {fi.get('channels')} ch, {_mmss(fi.get('decoded_duration', 0))}")
    lv = pp["level"]
    L.append(f"- loudness {lv['integrated']:.1f} LUFS, true peak {lv['true_peak']:.1f} dBTP" if lang == "en" else
             f"- громкость {lv['integrated']:.1f} LUFS, пик {lv['true_peak']:.1f} dBTP")
    for w in pp.get("windows", []):
        L.append(f"- {_mmss(w['start'])}–{_mmss(w['end'])}: SRMR {w.get('srmr', float('nan')):.1f}, "
                 f"DNSMOS SIG {w['sig']:.2f} / BAK {w['bak']:.2f} / OVRL {w['ovrl']:.2f}")
    L += ["", f"## {t['done']}", "", f"{t['route']}: {pl['explain_ru' if lang == 'ru' else 'explain']}", ""]
    for e in doc.get("events", []):
        if e.startswith(("Fix:", "  light", "Probe", "Session", "Best", "Sidon")) or " fills " in e:
            L.append(f"- {e}")
    if pl.get("warnings"):
        L += ["", f"**{t['warn']}:**"] + [f"- {w}" for w in pl["warnings"]]
    L += ["", f"## {t['check']}", "", f"| {t['gate']} | {t['status']} | {t['result']} |", "|---|---|---|"]
    for g in v["gates"]:
        L.append(f"| {_GATE[lang].get(g['name'], g['name'])} | {_STATUS[lang][g['status']]} | {g['summary']} |")
    q = next((g for g in v["gates"] if g["name"] == "not_worse"), None)
    if q and q["details"].get("windows"):
        L += ["", f"| {t['window']} | SRMR {t['before']} → {t['after']} | OVRL {t['before']} → {t['after']} | BAK {t['before']} → {t['after']} |",
              "|---|---|---|---|"]
        for r in q["details"]["windows"]:
            L.append(f"| {_mmss(r['start'])}–{_mmss(r['end'])} | {r['in']['srmr']:.1f} → {r['out']['srmr']:.1f} | "
                     f"{r['in']['ovrl']:.2f} → {r['out']['ovrl']:.2f} | {r['in']['bak']:.2f} → {r['out']['bak']:.2f} |")
    L += ["", f"## {t['cut']}", ""]
    if doc.get("trimmed"):
        for c in doc["trimmed"]:
            txt = f" «{c['recognised_text']}»" if c.get("recognised_text") else ""
            L.append(f"- {c['where']}: {_mmss(c['from_s'])}–{_mmss(c['to_s'])} ({c['duration_s']:.1f} s), "
                     f"speech {100 * c['speech_ratio']:.0f}%{txt}")
    else:
        L.append(t["nocut"])
    L += ["", f"## {t['unverified']}", "", f"- {t['ear']}"]
    for g in v["gates"]:
        if g["status"] in ("unverified", "skipped"):
            L.append(f"- {_GATE[lang].get(g['name'], g['name'])}: {g['summary']}")
    L += ["", f"## {t['files']}", ""]
    L.append(f"- `{Path(doc['output']).name}`")
    for k, p in (doc.get("files") or {}).items():
        if isinstance(p, str) and p.endswith((".wav", ".png")):
            L.append(f"- `{Path(p).name}` ({k})")
    return "\n".join(L) + "\n"


def render_refusal(doc: dict, lang: str = "en") -> str:
    t = _T.get(lang, _T["en"])
    return f"# {t['refused']}: {Path(doc['input']).name}\n\n{doc['reason']}\n"
