"""The chunk gluing of the Sidon engine is sample-exact over an hour (no model needed).

A stand-in decoder reproduces the two properties of the real one that broke timing: it returns
`frames * 960 - 26` samples per call and its output sits LATENCY_48K samples late. Every 20 ms
frame of the input carries a unique value; the output must reproduce the value of each frame at
exactly the right samples, over a full hour (37 chunk seams).
"""

import numpy as np
import soundfile as sf
import torch
from voice_rescue.engines import sidon


class Pre:
    def __call__(self, padded, sampling_rate, return_tensors):
        x = torch.from_numpy(np.asarray(padded, dtype=np.float32))
        frames = (len(x) - 320) // 320
        core = x[160: 160 + frames * 320].view(frames, 320)
        f = core.mean(dim=1, keepdim=True).repeat(1, 160)
        return {"input_features": f[None]}


class Fe:
    def __call__(self, feats):
        return {"last_hidden_state": feats}


class Dec:
    def __call__(self, f):  # (1, 160, T) -> T*960 - 26 samples, LATENCY_48K samples late
        v = f[0, 0]
        y = v.repeat_interleave(960)
        lat = sidon.LATENCY_48K
        y = torch.cat([torch.zeros(lat), y])[: len(y)]
        return y[: len(y) - sidon.DECODER_TAIL_LOSS][None]


def frame_values(n_frames):
    rng = np.random.default_rng(0)
    return rng.uniform(-0.5, 0.5, n_frames).astype(np.float32)


def test_hour_is_sample_exact(tmp_path):
    n_frames = 3600 * 50  # one hour of 20 ms frames
    v = frame_values(n_frames)
    x16 = np.repeat(v, 320)
    eng = sidon.Sidon(models=(Fe(), Dec(), Pre()))
    out = eng.restore(x16, tmp_path / "out.wav", target=len(x16) * 3)
    info = sf.info(str(out))
    assert info.frames == len(x16) * 3, "output length must be exactly 3x the 16 kHz input"
    expected_vals = v
    worst = 0.0
    with sf.SoundFile(str(out)) as fh:
        pos = 0
        while True:
            blk = fh.read(960 * 5000, dtype="float32")
            if not len(blk):
                break
            idx = (pos + np.arange(len(blk))) // 960
            exp = expected_vals[idx]
            worst = max(worst, float(np.max(np.abs(blk - exp))))
            pos += len(blk)
    # crossfades blend two identical decodes, so the result is exact up to float rounding
    assert worst < 1e-5, f"misplaced samples, max error {worst}"


def test_naive_gluing_drifts():
    """The reference demo recipe (glue chunk outputs, [:-960]) loses 26 samples per 96 s chunk."""
    per_chunk = 4801 * 960 - 26 - 960
    assert 4800 * 960 - per_chunk == 26
    assert abs(26 / (4800 * 960) * 1e6 - 5.64) < 0.01  # ppm, the drift measured on real recordings


def test_short_input_and_odd_length(tmp_path):
    v = frame_values(77)
    x16 = np.repeat(v, 320)[:-123]  # not a multiple of the frame size
    eng = sidon.Sidon(models=(Fe(), Dec(), Pre()))
    target = 73_421
    out = eng.restore(x16, tmp_path / "o.wav", target=target)
    y, sr = sf.read(str(out), dtype="float32")
    assert sr == 48000 and len(y) == target
    idx = np.arange(len(y)) // 960
    assert np.max(np.abs(y[: 76 * 960] - v[idx[: 76 * 960]])) < 1e-5
