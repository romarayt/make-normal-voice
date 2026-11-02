"""Every gate must fail on its own broken input, and pass on a correct one.

Inputs are open speech clips (LibriTTS-R test-clean, CC BY 4.0, see fixtures/ATTRIBUTION.md).
The "good" output is the input through the mix stage; each broken output damages exactly one thing:
added echo, a 50 ms shift, +3 LU, one replaced word, another person's voice, a click at a seam.
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from voice_rescue import asr, install, master, verify
from voice_rescue.audio import load_mono, write_wav

FIX = Path(__file__).parent / "fixtures"
SR = 48000

needs_models = pytest.mark.skipif(bool(install.missing(["dnsmos", "ecapa", "whisper-small"])),
                                  reason="gate models not installed (voice-rescue install dnsmos ecapa whisper-small)")


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    d = tmp_path_factory.mktemp("gates")
    src = FIX / "speaker_a.flac"
    good = d / "good.wav"
    master.master(src, good)
    return src, good, d


def stereo(x):
    return np.stack([x, x], 1)


def gates_of(src, out, **kw):
    res = verify.verify(src, out, **kw)
    return {g["name"]: g for g in res["gates"]}


@needs_models
def test_good_output_passes_the_damage_gates(pair):
    src, good, _ = pair
    g = gates_of(src, good, seams=[20.0])
    for name in ("not_worse", "words", "speaker", "sync", "loudness", "artifacts"):
        assert g[name]["status"] == "pass", (name, g[name]["summary"])


@needs_models
def test_echo_fails_not_worse(pair):
    src, good, d = pair
    x = load_mono(good, SR)
    rng = np.random.default_rng(1)
    t = np.arange(int(0.8 * SR)) / SR
    rir = rng.standard_normal(len(t)) * np.exp(-6.9 * t / 0.7)  # RT60 0.7 s
    rir[0] = 1.0
    y = np.convolve(x, rir)[: len(x)]
    y *= np.max(np.abs(x)) / np.max(np.abs(y))
    bad = write_wav(d / "echo.wav", stereo(y), SR)
    g = gates_of(src, bad, skip=("words", "speaker"))
    assert g["not_worse"]["status"] == "fail", g["not_worse"]["summary"]


@needs_models
def test_shift_50ms_fails_sync(pair):
    src, good, d = pair
    x = load_mono(good, SR)
    k = int(0.050 * SR)
    y = np.concatenate([np.zeros(k, np.float32), x[:-k]])
    bad = write_wav(d / "shift.wav", stereo(y), SR)
    g = gates_of(src, bad, skip=("words", "speaker"))
    assert g["sync"]["status"] == "fail", g["sync"]["summary"]
    assert abs(g["sync"]["details"]["median_ms"] - 50) < 2


@needs_models
def test_plus_3lu_fails_loudness(pair):
    src, good, d = pair
    x = load_mono(good, SR) * 10 ** (3 / 20)
    bad = write_wav(d / "loud.wav", stereo(x), SR)
    g = gates_of(src, bad, skip=("words", "speaker"))
    assert g["loudness"]["status"] == "fail", g["loudness"]["summary"]


@needs_models
def test_replaced_word_fails_words(tmp_path):
    src = FIX / "speaker_a_short.flac"
    good = tmp_path / "good.wav"
    master.master(src, good)
    x16 = load_mono(src, 16000)
    _, words, _ = asr.transcribe(x16, words=True, language="en")
    donor_x = load_mono(FIX / "speaker_a.flac", SR, start=30.0, duration=10.0)
    _, dwords, _ = asr.transcribe(load_mono(FIX / "speaker_a.flac", 16000, start=30.0, duration=10.0),
                                  words=True, language="en")
    # a confidently recognised content word, replaced by a different word of the same speaker
    # and of about the same length (no stretching: the substitute must stay intelligible)
    target = max((w for w in words[1:-1] if len(w.text.strip(".,!?")) >= 5), key=lambda w: w.prob)
    texts = {w.text.lower().strip(".,!?") for w in words}
    dur = target.end - target.start
    cands = [w for w in dwords if len(w.text.strip(".,!?")) >= 4 and w.text.lower().strip(".,!?") not in texts]
    donor = min(cands, key=lambda w: abs((w.end - w.start) - dur))
    x = load_mono(good, SR)
    a, b = int(target.start * SR), int(target.end * SR)
    piece = donor_x[int(donor.start * SR): int(donor.end * SR)]
    piece = np.pad(piece, (0, max(0, (b - a) - len(piece))))[: b - a].astype(np.float32)
    piece *= np.sqrt(np.mean(x[a:b] ** 2) / (np.mean(piece ** 2) + 1e-12))
    f = int(0.005 * SR)
    r = np.linspace(0, 1, f, dtype=np.float32)
    y = x.copy()
    y[a:b] = piece
    y[a:a + f] = x[a:a + f] * (1 - r) + piece[:f] * r
    y[b - f:b] = piece[-f:] * (1 - r) + x[b - f:b] * r
    bad = write_wav(tmp_path / "word.wav", stereo(y), SR)
    w_good = verify.gate_words(list(verify._windows_pairs(load_mono(src), load_mono(good), [(0.0, len(x16) / 16000)], verify.Mapping())), language="en")
    w_bad = verify.gate_words(list(verify._windows_pairs(load_mono(src), load_mono(bad), [(0.0, len(x16) / 16000)], verify.Mapping())), language="en")
    assert w_good.status == "pass", w_good.summary
    assert w_bad.status == "fail", (w_bad.summary, target.text, donor.text)


@needs_models
def test_other_voice_fails_speaker(pair):
    src, good, d = pair
    x = load_mono(good, SR)
    other = load_mono(FIX / "speaker_b.flac", SR)
    reps = int(np.ceil(len(x) / len(other)))
    y = np.tile(other, reps)[: len(x)]
    y *= np.sqrt(np.mean(x ** 2) / np.mean(y ** 2))
    bad = write_wav(d / "voice.wav", stereo(y), SR)
    g = gates_of(src, bad, skip=("words",))
    assert g["speaker"]["status"] == "fail", g["speaker"]["summary"]


@needs_models
def test_click_at_seam_fails_artifacts(pair):
    src, good, d = pair
    x = load_mono(good, SR)
    k = int(20.0 * SR)
    y = x.copy()
    y[k] += 0.6
    y[k + 1] -= 0.6
    bad = write_wav(d / "click.wav", stereo(y), SR)
    g = gates_of(src, bad, skip=("words", "speaker"), seams=[20.0])
    assert g["artifacts"]["status"] == "fail", g["artifacts"]["summary"]
    assert any(abs(r["seam_s"] - 20.0) < 0.01 for r in g["artifacts"]["details"]["seam_clicks"]["bad"])


@needs_models
def test_background_jump_at_seam_fails_artifacts(tmp_path):
    """Two parts glued with different room tone (e.g. a clean mic next to a restored part)."""
    from scipy import signal

    rng = np.random.default_rng(3)
    x = load_mono(FIX / "speaker_a.flac", SR)
    room = signal.lfilter([0.0499, -0.0960, 0.0506, -0.0044], [1, -2.4950, 2.0173, -0.5222],
                          rng.standard_normal(len(x))).astype(np.float32)
    room *= 10 ** (-58 / 20) / np.sqrt(np.mean(room ** 2))       # a quiet, continuous room tone
    src = write_wav(tmp_path / "in.wav", x + room, SR)
    good = tmp_path / "good.wav"
    master.master(src, good)
    y = load_mono(good, SR)
    k = int(35.0 * SR)
    hiss = rng.standard_normal(len(y) - k).astype(np.float32) * 10 ** (-40 / 20)
    y[k:] += hiss                                                 # the other part's room is 15+ dB louder
    bad = write_wav(tmp_path / "jump.wav", stereo(y), SR)
    ok = gates_of(src, good, skip=("words", "speaker"), seams=[35.0])
    g = gates_of(src, bad, skip=("words", "speaker"), seams=[35.0])
    assert ok["artifacts"]["status"] == "pass", ok["artifacts"]["summary"]
    assert g["artifacts"]["status"] == "fail", g["artifacts"]["summary"]
    assert g["artifacts"]["details"]["seam_jumps"]


def test_fixture_lengths():
    for n in ("speaker_a.flac", "speaker_b.flac", "speaker_a_short.flac"):
        assert sf.info(str(FIX / n)).duration > 3
