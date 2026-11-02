"""Speech recognition for checks only: did the words survive, and where does speech end.

faster-whisper invents text on quiet stretches ("Subtitles by ...", "To be continued..."). Guards:
VAD on, `condition_on_previous_text=False`, no prompt/glossary, and every recognised segment must
sit over audio that is actually louder than the pause floor.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .install import component_path

SR = 16000


@dataclass
class Word:
    text: str
    start: float
    end: float
    prob: float = 1.0


@lru_cache(maxsize=2)
def _model(name: str):
    from faster_whisper import WhisperModel

    return WhisperModel(str(component_path(f"whisper-{name}")), device="cpu", compute_type="int8")


def available(name: str) -> bool:
    try:
        component_path(f"whisper-{name}")
        return True
    except Exception:  # noqa: BLE001
        return False


def _energy_ok(x16: np.ndarray, a: float, b: float, floor_db: float) -> bool:
    seg = x16[int(a * SR): int(b * SR)]
    if len(seg) < SR // 20:
        return False
    hop = SR // 50
    n = len(seg) // hop
    if n == 0:
        return False
    e = 10 * np.log10((seg[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)
    return float(np.percentile(e, 90)) > floor_db + 6.0


def transcribe(x16: np.ndarray, *, model: str = "small", language: str | None = None, beam_size: int = 5,
               words: bool = False) -> tuple[str, list[Word], str]:
    """Return (text, words, language). Segments over silence are dropped as hallucinations."""
    m = _model(model)
    x16 = np.asarray(x16, dtype=np.float32)
    hop = SR // 50
    n = len(x16) // hop
    e = 10 * np.log10((x16[: n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12) if n else np.array([-120.0])
    floor_db = float(np.percentile(e, 10))
    segs, info = m.transcribe(x16, language=language, beam_size=beam_size, vad_filter=True,
                              condition_on_previous_text=False, word_timestamps=words,
                              temperature=0.0, without_timestamps=False)
    texts, ws = [], []
    for s in segs:
        if s.no_speech_prob > 0.6 and s.avg_logprob < -1.0:
            continue
        if not _energy_ok(x16, s.start, s.end, floor_db):
            continue
        texts.append(s.text.strip())
        if words and s.words:
            ws += [Word(w.word.strip(), w.start, w.end, float(w.probability)) for w in s.words]
    return " ".join(texts).strip(), ws, info.language


def tokens(text: str) -> list[str]:
    t = unicodedata.normalize("NFKC", text).lower().replace("ё", "е")
    return re.findall(r"\w+", t)


def wer(ref: str, hyp: str) -> float:
    r, h = tokens(ref), tokens(hyp)
    if not r:
        return 0.0 if not h else 1.0
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return float(d[len(h)] / len(r))


def _ratio(a: str, b: str) -> float:
    """Character similarity 0..1 (so that a changed ending is not a lost word)."""
    if a == b:
        return 1.0
    d = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev, d[j] = d[j], cur
    return 1.0 - d[len(b)] / max(len(a), len(b), 1)


def _align(r: list[str], h: list[str]) -> list[tuple[int, int | None]]:
    """Levenshtein alignment: for every reference index, the matched hypothesis index or None."""
    n, m = len(r), len(h)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    i, j, out = n, m, []
    while i > 0 and j > 0:
        if D[i, j] == D[i - 1, j - 1] + (r[i - 1] != h[j - 1]):
            out.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif D[i, j] == D[i - 1, j] + 1:
            out.append((i - 1, None))
            i -= 1
        else:
            j -= 1
    while i > 0:
        out.append((i - 1, None))
        i -= 1
    return out[::-1]


def _script(t: str) -> str:
    if any("\u0400" <= ch <= "\u04ff" for ch in t):
        return "cyrillic"
    if any("a" <= ch <= "z" for ch in t):
        return "latin"
    return "other"


def confident_loss(ref_words: list[Word], hyp_text: str, min_prob: float = 0.8, min_len: int = 3,
                   reach: int = 4) -> tuple[float, int, list[str]]:
    """Share of confidently recognised input words that are missing or changed in the output.

    Garbled stretches of a bad input (low recogniser confidence) are left out: the restored output
    may be recognised *better* there, which is not a changed word. Left out as well: numbers (one
    run writes "100", the next "сто") and words in another script than most of the text ("Елип" vs
    "ELIP"). A word counts as kept if a word with mostly the same letters is within `reach`
    positions of where the alignment puts it (one inserted word must not cascade into misses).
    """
    toks = [(tokens(w.text), w.prob) for w in ref_words]
    r, probs = [], []
    for tt, p in toks:
        for t in tt:
            r.append(t)
            probs.append(p)
    h = tokens(hyp_text)
    if not r:
        return 0.0, 0, []
    scripts = [_script(t) for t in r]
    main = max(set(scripts), key=scripts.count)
    al = _align(r, h)
    pos, last = [], 0
    for _, j in al:
        if j is not None:
            last = j
        pos.append(last)
    conf = [(i, pos[k]) for k, (i, _) in enumerate(al)
            if probs[i] >= min_prob and len(r[i]) >= min_len and not any(c.isdigit() for c in r[i])
            and scripts[i] == main]
    lost = [r[i] for i, p in conf
            if not any(_ratio(r[i], h[q]) >= 0.6 for q in range(max(0, p - reach), min(len(h), p + reach + 1)))]
    return (len(lost) / len(conf) if conf else 0.0), len(conf), lost
