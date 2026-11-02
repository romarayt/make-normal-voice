"""From a passport to a processing route, explained in one line per track."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from . import thresholds as T
from .analyze import summary


@dataclass
class Plan:
    name: str
    track_class: str
    status: str                     # proven | partly proven | hypothesis
    steps: list[str]
    reasons: list[str]
    warnings: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)

    def line(self, lang: str = "en") -> str:
        arrow = " → ".join(_STEP_NAMES[lang].get(s, s) for s in self.steps)
        return f"{self.name}: {', '.join(self.reasons) or _CLASS_NAMES[lang][self.track_class]} → {arrow}"

    def to_dict(self):
        d = asdict(self)
        d["explain"] = self.line("en")
        d["explain_ru"] = self.line("ru")
        return d


_STEP_NAMES = {
    "en": {"sidon": "Sidon", "master": "mix", "level_parts": "level the parts", "match": "match timbre",
           "declip": "declip", "hum_notch": "hum notch", "trim": "trim", "repair": "repair container"},
    "ru": {"sidon": "Sidon", "master": "сведение", "level_parts": "выровнять части", "match": "подогнать тембр",
           "declip": "убрать клиппинг", "hum_notch": "убрать фон сети", "trim": "обрезка", "repair": "починить файл"},
}
_CLASS_NAMES = {
    "en": {"far_mic_echo": "far mic, room echo", "laptop_mic": "built-in laptop mic", "close_noisy": "close mic, noisy",
           "close_clean": "clean close mic", "unknown": "speech", "unreadable": "unreadable"},
    "ru": {"far_mic_echo": "дальний микрофон, эхо", "laptop_mic": "встроенный микрофон ноутбука",
           "close_noisy": "близкий микрофон, шум", "close_clean": "чистый близкий микрофон", "unknown": "речь",
           "unreadable": "файл не читается"},
}


def _fmt(v: float) -> str:
    return f"{v:.1f}"


def plan(passport: dict, *, name: str | None = None, glue_with_sidon: bool = False,
         experimental: dict | None = None) -> Plan:
    exp = {**T.EXPERIMENTAL_DEFAULTS, **(experimental or {})}
    fi = passport["file"]
    name = name or fi["path"].rsplit("/", 1)[-1]
    if not fi.get("readable"):
        return Plan(name, "unreadable", "hypothesis", ["repair"], ["no decodable audio stream"],
                    ["the file could not be read; make-normal-voice will try a repair remux and refuse if that fails"])
    s = summary(passport)
    srmr, ovrl, bak = s.get("srmr", 99.0), s.get("ovrl", 5.0), s.get("bak", 5.0)
    lvl = passport["level"]
    reasons, warnings, steps, params = [], [], [], {}

    if srmr < T.FAR_MIC_SRMR or ovrl < T.FAR_MIC_OVRL:
        cls, status = "far_mic_echo", "proven"
        reasons.append(f"echo (SRMR {_fmt(srmr)}, OVRL {ovrl:.2f})")
    elif srmr < T.LAPTOP_SRMR and ovrl < T.LAPTOP_OVRL:
        cls, status = "laptop_mic", "proven"
        reasons.append(f"laptop-mic sound (SRMR {_fmt(srmr)}, OVRL {ovrl:.2f})")
    elif bak < T.CLEAN_BAK:
        cls, status = "close_noisy", "proven"
        reasons.append(f"background noise (BAK {bak:.2f})")
    elif ovrl >= T.CLEAN_OVRL and srmr >= T.CLEAN_SRMR:
        cls, status = "close_clean", "proven"
        reasons.append(f"already clean (OVRL {ovrl:.2f}, BAK {bak:.2f}, SRMR {_fmt(srmr)})")
    else:
        cls, status = "unknown", "partly proven"
        reasons.append(f"OVRL {ovrl:.2f}, SRMR {_fmt(srmr)}")

    if lvl["integrated"] < -30:
        reasons.append(f"quiet ({lvl['integrated']:.0f} LUFS)")
    dmg = passport.get("damage", {})
    if dmg.get("clip_ratio", 0) > T.CLIP_RATIO or dmg.get("flat_top_samples", 0) > T.FLAT_TOPS:
        reasons.append(f"clipping ({100 * dmg['clip_ratio']:.2f}% samples)")
        if exp["declip"]:
            steps.append("declip")
        else:
            warnings.append("clipping found; the declip branch is experimental and off (see README benchmark)")
    hum = passport.get("noise", {}).get("hum", {})
    if hum.get("freq"):
        reasons.append(f"{hum['freq']} Hz hum")
        if exp["hum_notch"]:
            steps.append("hum_notch")
            params["hum_freq"] = hum["freq"]
        else:
            warnings.append("mains hum found; Sidon is relied on to remove it (notch branch is experimental)")
    band = passport.get("band", {})
    if band.get("class") in ("phone", "narrow"):
        reasons.append(f"narrow band (to {band['cutoff_hz']} Hz)")
    if passport.get("content", {}).get("music_suspected"):
        warnings.append("music or other tonal sound under the voice is suspected; restoration may remove "
                        "or garble it (no music separation in v0.2)")
    if dmg.get("digital_zero_count", 0):
        warnings.append(f"{dmg['digital_zero_count']} digital dropouts; without another source they stay as they are")
    st = passport.get("stereo")
    if st:
        if st.get("dead_channel") is not None:
            warnings.append(f"channel {st['dead_channel'] + 1} is dead or much quieter; the other one is used")
            params["channel"] = 1 - st["dead_channel"]
        elif st.get("inverted"):
            warnings.append("channels are polarity-inverted; using the first channel only")
            params["channel"] = 0

    if cls == "close_clean" and not glue_with_sidon:
        steps.append("master")
    else:
        steps += ["sidon"]
        parts = lvl.get("parts", [])
        if len(parts) > 1:
            spread = max(p["speech_lufs"] for p in parts) - min(p["speech_lufs"] for p in parts)
            if spread > T.PARTS_LU:
                reasons.append(f"parts differ by {spread:.1f} LU")
                steps.append("level_parts")
                params["parts"] = parts
        steps.append("master")
    if cls == "close_clean" and glue_with_sidon:
        reasons.append("restored anyway to match parts that went through Sidon")
    return Plan(name, cls, status, steps, reasons, warnings, params)
