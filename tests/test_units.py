"""Fast unit tests: loudness, mixing chain, timbre EQ delay, alignment and drift, routing, file names."""

import json
import re
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from voice_rescue import align, ffmpeg, loudness, master, match, route
from voice_rescue.audio import load_channel, load_mono, write_wav

FIX = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "voice-rescue"


def speech(sr=48000, start=0.0, dur=20.0):
    return load_mono(FIX / "speaker_a.flac", sr, start=start, duration=dur)


def test_numpy_loudness_matches_ffmpeg():
    x = speech()
    assert abs(loudness.integrated(x, 48000) - ffmpeg.loudness_array(x, 48000).integrated) < 0.15


def test_dual_mono_reads_3_lu_louder(tmp_path):
    x = speech()
    m = write_wav(tmp_path / "m.wav", x, 48000)
    s = write_wav(tmp_path / "s.wav", np.stack([x, x], 1), 48000)
    assert abs(ffmpeg.loudness(s).integrated - ffmpeg.loudness(m).integrated - 3.0) < 0.15


def test_master_hits_target_without_moving_time(tmp_path):
    x = speech() * 0.05  # quiet input, like a far camera mic
    k = 48000 * 7
    x[k] += 0.04  # a marker transient
    src = write_wav(tmp_path / "in.wav", x, 48000, subtype="FLOAT")
    out = tmp_path / "out.wav"
    r = master.master(src, out)
    info = sf.info(str(out))
    assert (info.samplerate, info.channels, info.subtype) == (48000, 2, "PCM_24")
    assert info.frames == len(x)
    m = ffmpeg.loudness(out)
    assert abs(m.integrated + 16.0) <= 0.1 and m.true_peak <= -1.0, r
    y = load_channel(out)
    o = align.local_offset(x, y, 48000, 10.0, win=8.0, search=0.01, hop_s=0.0005)
    assert abs(o.seconds) * 1000 < 0.3


def test_no_compressor_in_the_mix():
    chain = master.PRE + master.limiter(0.84)
    for word in ("acompressor", "compand", "dynaudnorm", "loudnorm", "extrastereo", "stereowiden"):
        assert word not in chain
    assert "latency=1" in master.limiter(0.84)


def test_fir_eq_has_no_delay_and_matches_timbre():
    sr = 48000
    rng = np.random.default_rng(0)
    x = speech()
    tilt = np.interp(np.log10(match.CENTERS), np.log10([100, 1000, 3000, 8000, 20000]), [3, 0, -4, -5, -5])
    colored = match.apply_fir(x, match.design_fir(match.CENTERS, tilt, sr))  # a duller, boomier mic
    y, _ = match.match(x, colored, sr, passes=2)
    _, d = match.residual(x, y, sr, lo=100, hi=8000)
    assert np.max(np.abs(d)) < 1.0
    imp = np.zeros(sr, np.float32)
    imp[sr // 2] = 1.0
    h = match.design_fir(match.CENTERS[:20], rng.uniform(-6, 6, 20), sr)
    z = match.apply_fir(imp, h)
    assert int(np.argmax(np.abs(z))) == sr // 2


def test_drift_and_offset_are_recovered():
    sr = 16000
    a = load_mono(FIX / "speaker_a.flac", sr)
    a = np.concatenate([a, load_mono(FIX / "speaker_b.flac", sr), a])
    ppm, off = 6.1, 1.234
    b = align.stretch(a, sr, -ppm)  # b's clock runs slow against a by 6.1 ppm
    b = b[int(off * sr):]           # and b starts 1.234 s into a
    d = align.measure_drift(a, b, sr, n_points=10, min_span=100)
    assert abs(d.offset - off) < 0.0005
    assert abs(d.ppm - ppm) < 1.0
    fixed = align.place(align.stretch(b, sr, d.ppm), sr, d.offset, len(a))
    for c in (30.0, 90.0, 150.0):
        o = align.local_offset(a, fixed, sr, c, win=10, search=0.02, guess=0.0)
        assert abs(o.seconds) < 0.001
    # over a span shorter than min_span the drift is not fitted (noise would become a fake drift)
    assert align.measure_drift(a, b, sr, n_points=10).ppm == 0.0


def test_alignment_finds_the_direct_sound_through_echo():
    """A far, echoey copy of a close recording: GCC-PHAT lands on the direct path, the envelope does not."""
    from scipy import signal

    sr = 16000
    lav = load_mono(FIX / "speaker_a.flac", sr)
    rng = np.random.default_rng(2)
    t = np.arange(int(0.6 * sr)) / sr
    tail = rng.standard_normal(len(t)) * np.exp(-6.9 * t / 0.6)
    tail[0] = 0.0
    rir = tail * np.sqrt(10 ** 0.5 / np.sum(tail ** 2))
    rir[0] = 1.0
    cam = np.concatenate([np.zeros(int(2.345 * sr), np.float32), signal.fftconvolve(lav, rir)[: len(lav)].astype(np.float32)])
    d = align.measure_drift(cam, lav, sr)
    assert abs(d.offset - 2.345) < 0.0002
    assert abs(align.coarse_offset(cam, lav, sr).seconds - 2.345) > 0.004  # the envelope is pulled late


def test_same_session_score():
    a = load_mono(FIX / "speaker_a.flac")
    same, o = align.same_session(a, a[16000 * 5: 16000 * 50])
    other, o2 = align.same_session(a, load_mono(FIX / "speaker_b.flac"))
    assert same and o.score > 0.9 and abs(o.seconds - 5.0) < 0.02
    assert not other and o2.score < 0.5


def _pp(srmr, ovrl, bak, lufs=-20.0, parts=None, clip=0.0):
    w = {"srmr": srmr, "ovrl": ovrl, "bak": bak, "sig": 3.5, "start": 0, "end": 60}
    return {"file": {"path": "x.wav", "readable": True}, "windows": [w, w, w],
            "level": {"integrated": lufs, "parts": parts or [{"start": 0, "end": 100, "speech_lufs": -20}]},
            "damage": {"clip_ratio": clip, "flat_top_samples": 0, "digital_zero_count": 0},
            "noise": {"hum": {"freq": None}}, "band": {"class": "full", "cutoff_hz": 20000}, "content": {}}


def test_routes():
    assert route.plan(_pp(5.9, 2.53, 3.3, lufs=-40)).steps == ["sidon", "master"]
    p = route.plan(_pp(5.9, 2.53, 3.3, lufs=-40))
    assert p.track_class == "far_mic_echo" and "quiet" in p.line()
    assert route.plan(_pp(11.0, 2.9, 3.8)).track_class == "laptop_mic"
    assert route.plan(_pp(14.0, 3.4, 4.1)).steps == ["master"]
    assert route.plan(_pp(14.0, 3.4, 4.1), glue_with_sidon=True).steps == ["sidon", "master"]
    two = [{"start": 0, "end": 120, "speech_lufs": -26}, {"start": 120, "end": 900, "speech_lufs": -20}]
    assert "level_parts" in route.plan(_pp(11.0, 2.9, 3.8, parts=two)).steps
    clip = route.plan(_pp(11.0, 2.9, 3.8, clip=0.01))
    assert "declip" not in clip.steps and any("clipping" in w for w in clip.warnings)


def test_ffmpeg_args_survive_odd_file_names(tmp_path):
    x = speech(dur=3.0)
    for name in ("урок 01 — эхо.wav", "a=b c's \"q\".wav", "=echo $OFF:linear.wav"):
        p = write_wav(tmp_path / name, x, 48000)
        y = load_mono(p, 16000)
        assert len(y) == 3 * 16000


def test_skill_frontmatter_follows_the_spec():
    text = (SKILL / "SKILL.md").read_text()
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert m
    import yaml  # noqa: F401 - pyyaml comes with transformers
    meta = __import__("yaml").safe_load(m.group(1))
    assert meta["name"] == SKILL.name == "voice-rescue"
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", meta["name"]) and len(meta["name"]) <= 64
    assert 0 < len(meta["description"]) <= 1024
    assert len(meta.get("compatibility", "")) <= 500
    assert all(isinstance(v, str) for v in meta.get("metadata", {}).values())
    assert len(text.splitlines()) < 500
    for ref in re.findall(r"`(references/[\w.-]+)`", text):
        assert (SKILL / ref).exists(), ref


def test_plugin_manifests():
    pj = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text())
    mj = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text())
    assert pj["name"] == "voice-rescue"
    assert mj["plugins"][0]["name"] == "voice-rescue" and mj["plugins"][0]["source"] == "./"
    assert pj["version"] == mj["plugins"][0]["version"]


SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".vr-home", "e2e"}
TRACKED = sorted(p for p in ROOT.rglob("*") if p.is_file() and not SKIP_DIRS & set(p.parts)
                 and p.suffix in {".py", ".md", ".json", ".toml", ".yml", ".sh", ".ps1", ".txt", ""})


@pytest.mark.parametrize("path", TRACKED, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_private_paths(path):
    text = path.read_text(errors="ignore")
    assert not re.search(r"/Users/[a-z]|/home/[a-z]+/|C:\\\\Users", text), path


def test_srmr_matches_reference_algorithm():
    """Our band-by-band SRMR equals torchmetrics' (the reference port of SRMRpy) where it is installed."""
    tm = pytest.importorskip("torchmetrics.functional.audio.srmr")
    pytest.importorskip("torchaudio")
    import torch
    from voice_rescue import metrics

    for name, start in (("speaker_a.flac", 3.0), ("speaker_b.flac", 20.0)):
        x = metrics.prep(load_mono(FIX / name, 16000, start=start, duration=8.0))
        ref = float(tm.speech_reverberation_modulation_energy_ratio(torch.from_numpy(x), 16000))
        assert abs(metrics.srmr(x) - ref) < 1e-3, (name, metrics.srmr(x), ref)


def test_memory_guard_shortens_or_refuses(monkeypatch):
    from voice_rescue import memory

    monkeypatch.setattr(memory, "available_gb", lambda: 40.0)
    assert memory.pick_chunk_seconds(30.0) == 30.0
    monkeypatch.setattr(memory, "available_gb", lambda: 5.0)
    assert memory.pick_chunk_seconds(30.0) == 20.0
    monkeypatch.setattr(memory, "available_gb", lambda: 2.0)
    with pytest.raises(memory.NotEnoughMemory):
        memory.pick_chunk_seconds(30.0)
    with pytest.raises(memory.NotEnoughMemory):
        memory.check_during_run(min_gb=3.0)
