"""Natural route: WPE dereverberation and an EQ toward the restored timbre, no re-synthesis of the voice.

Also pins that a route the user forced is kept by the probe and the fix loop (it used to be swapped
for Sidon silently, so `--route master` delivered a Sidon track).
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import soxr
from make_normal_voice import align, match, metrics, pipeline, route, stages
from make_normal_voice.audio import load_mono
from make_normal_voice.engines import wpe
from scipy.signal import fftconvolve

FIX = Path(__file__).parent / "fixtures"
SR = 48000


def speech(dur=20.0, start=0.0):
    return load_mono(FIX / "speaker_a.flac", SR, start=start, duration=dur)


def corr(a, b):
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


def srmr48(x):
    return metrics.srmr(metrics.prep(soxr.resample(x, SR, 16000)))


def room(x, rt60=0.3, drr_db=3.0, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(int(0.8 * SR)) / SR
    tail = rng.standard_normal(len(t)) * np.exp(-6.9 * t / rt60)
    tail *= 10 ** (-drr_db / 20) / np.sqrt(np.sum(tail ** 2))
    rir = np.zeros(len(t))
    rir[0] = 1.0
    k = int(0.003 * SR)
    rir[k:] += tail[: len(rir) - k]
    return fftconvolve(x, rir)[: len(x)].astype(np.float32)


def test_wpe_removes_room_and_leaves_a_dry_voice_alone():
    dry = speech()
    wet = room(dry)
    y = wpe.dereverb(wet, chunk_s=8.0, overlap_s=1.0)
    assert len(y) == len(wet)
    assert srmr48(y) > srmr48(wet) + 1.0           # measured 8.6 -> 10.6
    assert corr(y, dry) > corr(wet, dry)
    yd = wpe.dereverb(dry, chunk_s=8.0, overlap_s=1.0)
    assert corr(yd, dry) > 0.99                    # it subtracts a prediction, it does not rebuild the voice
    assert abs(srmr48(yd) - srmr48(dry)) < 0.5


def test_wpe_keeps_time_and_chunks_join_smoothly():
    x = speech(dur=26.0) * 0.5
    k = SR * 7 + 123
    x[k] += 0.3                                    # a marker transient away from any seam
    y = wpe.dereverb(x, chunk_s=8.0, overlap_s=1.0)
    assert len(y) == len(x)
    o = align.local_offset(x, y, SR, 10.0, win=8.0, search=0.01, hop_s=0.0005)
    assert abs(o.seconds) * 1000 < 0.3
    assert int(np.argmax(np.abs(y[k - 2400:k + 2400]))) == 2400
    # on a reverberant voice, chunked and one-piece results agree across each seam: measured 3-4 %
    # with the 1 s crossfade, 13-15 % with hard cuts, 31 % on a 3 s last chunk (now merged)
    w = room(x)
    yc, whole = wpe.dereverb(w, chunk_s=8.0, overlap_s=1.0), wpe.dereverb_chunk(w)
    assert wpe.seams(len(w) / SR, chunk_s=8.0) == [8.0, 16.0]
    for s in wpe.seams(len(w) / SR, chunk_s=8.0):
        i = int(s * SR)
        a, b = yc[i - 2400:i + 2400], whole[i - 2400:i + 2400]
        assert np.sqrt(np.mean((a - b) ** 2)) < 0.08 * np.sqrt(np.mean(b ** 2)), s
    assert wpe.seams(150.0) == [60.0, 120.0] and wpe.seams(130.0) == [60.0]
    assert wpe.spans(130 * SR)[-1] == (58 * SR, 130 * SR)


def _wpe_reference(Y, taps=30, delay=2, iterations=3):
    """The same algorithm written plainly with einsum in complex128."""
    F, T = Y.shape
    Yt = np.zeros((F, T, taps), complex)
    for k in range(taps):
        Yt[:, delay + k:, k] = Y[:, : T - delay - k]
    X = Y
    for _ in range(iterations):
        p = np.abs(X) ** 2
        lam = np.maximum(p, 1e-10 * p.max() + 1e-20)
        A = Yt / lam[:, :, None]
        R = np.einsum("ftk,ftl->fkl", A, Yt.conj())
        P = np.einsum("ftk,ft->fk", A, Y.conj())
        R += (1e-6 * np.trace(R, axis1=1, axis2=2).real / taps)[:, None, None] * np.eye(taps)
        X = Y - np.einsum("fk,ftk->ft", np.linalg.solve(R, P[..., None])[..., 0].conj(), Yt)
    return X


def test_wpe_runs_in_double_precision():
    """In complex64 one 375 Hz bin of a real laptop chunk came out 60x too loud (DNSMOS SIG 3.60 -> 2.44)
    and the result changed from run to run; the solve must match a plain float64 reference."""
    rng = np.random.default_rng(3)
    F, T = 32, 3000
    amp = 10 ** rng.uniform(-6, 0, size=(1, T))           # 120 dB of frame power
    Y = (rng.standard_normal((F, T)) + 1j * rng.standard_normal((F, T))) * amp
    Y[:, 1:] += 0.6 * Y[:, :-1] * np.exp(1j * 0.3)       # some predictable "reverberation"
    ref = _wpe_reference(Y)
    got = wpe._wpe_bins(Y, wpe.TAPS, wpe.DELAY, wpe.ITERATIONS)
    err = np.sqrt(np.mean(np.abs(got - ref) ** 2) / np.mean(np.abs(ref) ** 2))
    assert err < 1e-8, err


def test_wpe_guard_returns_a_diverged_bin_to_the_input():
    rng = np.random.default_rng(4)
    Y = rng.standard_normal((4, 500)) + 1j * rng.standard_normal((4, 500))
    X = 0.5 * Y
    X[2] = 60.0 * Y[2]                                     # a blown-up solve
    wpe.guarded = 0
    out = wpe.guard(Y, X)
    assert np.allclose(out[2], Y[2]) and np.allclose(out[[0, 1, 3]], X[[0, 1, 3]]) and wpe.guarded == 1


def test_timbre_to_restored_matches_without_moving_time():
    x = speech()
    k = SR * 5 + 77
    x[k] += 0.3
    tilt = np.interp(np.log10(match.CENTERS), np.log10([100, 1000, 4000, 8000, 20000]), [0, 0, 6, 9, 9])
    bright = match.apply_fir(x, match.design_fir(match.CENTERS, tilt, SR))  # stands in for Sidon's output
    pairs = [(x[SR * a:SR * b], bright[SR * a:SR * b]) for a, b in ((1, 7), (10, 16))]
    y, log = stages.timbre_to_restored(x, pairs)
    _, d = match.residual(bright, y, SR, lo=100, hi=12000)
    assert np.max(np.abs(d)) < 1.0, d
    assert len(log) == 2 and len(y) == len(x)
    assert int(np.argmax(np.abs(y[k - 2400:k + 2400]))) == 2400


def _plan(parts=True):
    w = {"start": 0, "end": 60, "srmr": 9.3, "ovrl": 3.05, "bak": 3.6, "sig": 3.6}
    two = [{"start": 0, "end": 52, "speech_lufs": -26}, {"start": 52, "end": 900, "speech_lufs": -29}]
    pp = {"file": {"path": "x.wav", "readable": True}, "windows": [w, w, w],
          "level": {"integrated": -29, "parts": two if parts else [{"start": 0, "end": 900, "speech_lufs": -29}]},
          "damage": {"clip_ratio": 0, "flat_top_samples": 0, "digital_zero_count": 0},
          "noise": {"hum": {"freq": None}}, "band": {"class": "full", "cutoff_hz": 20000}, "content": {}}
    return route.plan(pp)


def test_forced_routes():
    assert _plan().steps == ["sidon", "level_parts", "master"]
    nat = route.force(_plan(), "natural")
    assert nat.steps == ["dereverb", "timbre_to_restored", "level_parts", "master"]
    assert "Sidon" not in nat.line("en") and "dereverb (WPE)" in nat.line("en")
    assert "убрать эхо (WPE)" in nat.line("ru")
    assert route.force(_plan(), "master").steps == ["master"]
    assert route.force(_plan(parts=False), "sidon").steps == ["sidon", "master"]
    assert route.force(_plan(), None).steps == ["sidon", "level_parts", "master"]
    with pytest.raises(ValueError):
        route.force(_plan(), "denoise")


@pytest.mark.parametrize("forced", ["master", "natural"])
def test_fix_loop_keeps_a_forced_route(tmp_path, monkeypatch, forced):
    run = pipeline.Run([FIX / "speaker_a_short.flac"], pipeline.Options(out_dir=tmp_path, force_route=forced,
                                                                         log=lambda m: None))

    def swapped(*a, **k):
        raise AssertionError("the fix loop re-ran the track on another route")

    monkeypatch.setattr(run, "_restore_track", swapped)
    pl = route.force(_plan(), forced)
    before = list(pl.steps)
    v = {"gates": [{"name": "quality", "status": "fail", "details": {}},
                   {"name": "not_worse", "status": "fail", "details": {"windows": []}}]}
    assert run._fix(v, FIX / "speaker_a_short.flac", {}, pl, np.zeros(10, np.float32), None, tmp_path / "f.wav",
                    None) is None
    assert pl.steps == before
    assert any("forced" in e for e in run.events)


def test_natural_eq_is_measured_on_the_right_stretches(tmp_path, monkeypatch):
    """Sidon runs only on the analysis windows; each restored stretch is paired with the same stretch of x."""
    src = FIX / "speaker_a.flac"
    full = load_mono(src, SR, start=0, duration=30.0)
    calls, seen = [], []

    def fake_sidon(path, *, target_len, start=None, duration=None, **kw):
        calls.append((start, duration, target_len))
        seg = full[int(round(start * SR)):int(round(start * SR)) + target_len]
        out = tmp_path / f"r{len(calls)}.wav"
        sf.write(out, seg * 2.0, SR, subtype="FLOAT")   # the "restored" stretch: the same audio, louder
        return out

    def capture(x, pairs, **kw):
        seen[:] = pairs
        return x, [{"bands_hz": [4000], "gain_db": [0.0]}]

    monkeypatch.setattr(stages, "sidon_restore", fake_sidon)
    monkeypatch.setattr(stages, "timbre_to_restored", capture)
    run = pipeline.Run([src], pipeline.Options(out_dir=tmp_path, log=lambda m: None))
    run._pp = {"windows": [{"start": 2.0, "end": 8.0}, {"start": 12.5, "end": 18.5}]}
    pl = route.force(_plan(parts=False), "natural")
    run._timbre_to_restored(src, pl, full.copy(), start=None, pre_af=None)
    assert [(a, d) for a, d, _ in calls] == [(2.0, 6.0), (12.5, 6.0)]
    assert all(n == 6 * SR for _, _, n in calls)
    for (seg, rest), (a, _, n) in zip(seen, calls):
        i = int(round(a * SR))
        assert len(seg) == n and np.allclose(seg, full[i:i + n]) and np.allclose(rest, 2.0 * seg, atol=1e-6)
    calls.clear()
    probe = full[SR * 4:SR * 24].copy()                 # a probe: x starts 4 s into the source
    run._timbre_to_restored(src, pl, probe, start=4.0, pre_af=None)
    assert calls == [(4.0, 20.0, 20 * SR)]
    seg, rest = seen[0]
    assert len(seg) == 20 * SR and np.allclose(seg, probe) and np.allclose(rest, 2.0 * seg, atol=1e-6)


def test_dereverb_cache_is_never_reused_across_engine_revisions(tmp_path, monkeypatch):
    """A complex64 result was cached once; a fixed engine must not pick it up again."""
    monkeypatch.setenv("MAKE_NORMAL_VOICE_HOME", str(tmp_path))
    runs = []
    monkeypatch.setattr(wpe, "dereverb", lambda x, **k: runs.append(1) or x * 0.5)
    x = np.random.default_rng(5).standard_normal(301 * SR).astype(np.float32) * 0.01
    logs = []
    stages.dereverb(x, log=logs.append)
    stages.dereverb(x, log=logs.append)
    assert len(runs) == 1 and any("cached" in m for m in logs)
    monkeypatch.setattr(wpe, "REVISION", "next")
    stages.dereverb(x, log=logs.append)
    assert len(runs) == 2
