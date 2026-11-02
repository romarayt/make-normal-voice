"""Tests with the real Sidon weights (skipped when they are not installed; run in the manual CI job).

* exact length and 1 ms sync on an open clip;
* chunk seams are inaudible: decoding with 10 s chunks (7 seams) matches one-chunk decoding;
* a clean recording is not made worse.
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from voice_rescue import align, install, metrics, verify
from voice_rescue.audio import load_mono
from voice_rescue.engines import sidon

FIX = Path(__file__).parent / "fixtures"
pytestmark = pytest.mark.skipif(bool(install.missing(["sidon-feature-extractor-cpu", "sidon-decoder-cpu", "dnsmos"])),
                                reason="Sidon weights not installed")


@pytest.fixture(scope="module")
def engine():
    return sidon.Sidon("cpu")


@pytest.fixture(scope="module")
def restored(engine, tmp_path_factory):
    src = FIX / "speaker_a.flac"
    x = sidon.prepare_input(src)
    info = sf.info(str(src))
    target = int(round(info.frames * 48000 / info.samplerate))
    out = engine.restore(x, tmp_path_factory.mktemp("s") / "one.wav", target=target)
    return src, out, target


def test_length_is_exact(restored):
    _, out, target = restored
    assert sf.info(str(out)).frames == target


def test_sync_within_1ms(restored):
    src, out, _ = restored
    a, b = load_mono(src), load_mono(out)
    offs = []
    for c in np.linspace(12, len(b) / 16000 - 12, 7):
        o = align.local_offset(a, b, 16000, c, win=16.0, search=0.04, guess=0.0)
        offs.append(-o.seconds * 1000)
    assert abs(float(np.median(offs))) <= 1.0, offs


def test_seams_match_single_chunk(engine, restored, tmp_path):
    src, out, target = restored
    x = sidon.prepare_input(src)
    many = engine.restore(x, tmp_path / "many.wav", target=target, chunk_frames=500)  # 10 s chunks
    y1, _ = sf.read(str(out), dtype="float32")
    y2, _ = sf.read(str(many), dtype="float32")
    assert len(y1) == len(y2)
    seams = [k * 500 * 960 for k in range(1, len(y2) // (500 * 960) + 1)]
    r = verify.seam_spikes(y2, 48000, [s / 48000 for s in seams])
    assert not r["bad"], r
    # the two decodes differ only as much as two contexts differ, and not more at the seams than elsewhere
    d = y1 - y2
    w = 960
    near = np.concatenate([d[s - w: s + w] for s in seams if s + w < len(d)])
    rms_near = float(np.sqrt(np.mean(near ** 2)))
    rms_all = float(np.sqrt(np.mean(d ** 2)))
    assert rms_near <= 3 * rms_all + 1e-6, (rms_near, rms_all)


def test_clean_input_not_made_worse(restored):
    src, out, _ = restored
    a, b = load_mono(src), load_mono(out)
    sa, sb = metrics.score(a[: 60 * 16000]), metrics.score(b[: 60 * 16000])
    assert sb["ovrl"] >= sa["ovrl"] - 0.05, (sa, sb)
    assert sb["srmr"] >= 9.0, (sa, sb)


def test_session_lavalier_onto_camera_timeline(tmp_path):
    """A clean 'lavalier' and an echoey, later-starting, drifting 'camera' of the same speech:
    the result must sit on the camera timeline to 1 ms and come from the lavalier."""
    from scipy import signal
    from voice_rescue import pipeline
    from voice_rescue.audio import write_wav

    lav = load_mono(FIX / "speaker_a.flac", 48000)
    rng = np.random.default_rng(2)
    t = np.arange(int(0.6 * 48000)) / 48000
    tail = rng.standard_normal(len(t)) * np.exp(-6.9 * t / 0.6)
    tail[0] = 0.0
    rir = tail * np.sqrt(10 ** 0.5 / np.sum(tail ** 2))  # direct-to-reverberant ratio -5 dB, RT60 0.6 s
    rir[0] = 1.0
    cam = signal.fftconvolve(lav, rir)[: len(lav)].astype(np.float32)
    cam = align.stretch(cam, 48000, 6.0)          # camera clock 6 ppm off
    cam = np.concatenate([np.zeros(int(2.345 * 48000), np.float32), cam])  # camera started earlier
    cam *= 0.3 / np.abs(cam).max()
    lav_p = write_wav(tmp_path / "lav.wav", lav, 48000)
    cam_p = write_wav(tmp_path / "cam.wav", cam, 48000)
    doc = pipeline.process([lav_p, cam_p], pipeline.Options(out_dir=tmp_path / "out", sync_to=cam_p, probe=False,
                                                            language="en"))
    out = Path(doc["output"])
    assert sf.info(str(out)).frames == len(cam)
    assert Path(doc["stages"]["session"][0]["path"]).name == "lav.wav"
    # where the lavalier speech truly sits on the camera timeline (dry, so the envelope is exact)
    truth = align.place(align.stretch(lav, 48000, 6.0), 48000, 2.345, len(cam))
    t16 = write_wav(tmp_path / "truth.wav", truth, 48000)
    a, b = load_mono(t16), load_mono(out)
    offs = [align.local_offset(a, b, 16000, c, win=16.0, search=0.03, guess=0.0).seconds * 1000
            for c in np.linspace(12, len(b) / 16000 - 12, 7)]
    assert abs(float(np.median(offs))) <= 1.0, offs


def test_windowed_decoder_equals_one_call(engine):
    """The decoder runs on 3 s windows to bound memory; the samples must equal one long call."""
    import torch

    x = sidon.prepare_input(FIX / "speaker_a.flac")[: 16000 * 9]
    n = len(x) // 320
    xp = np.concatenate([x, np.zeros(16000, np.float32)])
    f = engine._features(xp, 0, n + 20)
    with torch.inference_mode():
        full = engine.dec(f.transpose(1, 2)).view(-1).float().numpy()
    win = engine._decode(f, 0, n)
    assert len(win) == n * 960
    assert float(np.max(np.abs(full[: len(win)] - win))) < 1e-5
