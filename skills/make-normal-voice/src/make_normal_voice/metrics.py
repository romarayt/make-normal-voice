"""No-reference speech metrics: SRMR (dryness) and DNSMOS P.835 (SIG/BAK/OVRL).

* SRMR: speech-to-reverberation modulation energy ratio (the SRMRpy/torchmetrics algorithm, computed
  one band at a time to keep memory small). Higher is drier. Room echo inside the words pulls it down.
* DNSMOS P.835: Microsoft's model of listener ratings, 1-5. SIG = speech, BAK = background,
  OVRL = overall. The ONNX file is loaded from make-normal-voice's own model folder; the scoring code is
  a faithful port of the torchmetrics/Microsoft reference (same segmenting and polynomial mapping),
  so numbers are comparable with earlier measurements made with torchmetrics.

Both are computed on 16 kHz mono, peak-normalised to 0.5, on speech windows (not on pauses: a
quiet pause after a word says nothing about the echo inside the word).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .install import component_path

SR = 16000
DNSMOS_LEN = 9.01


@lru_cache(maxsize=1)
def _dnsmos_session():
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(str(component_path("dnsmos")), sess_options=opts, providers=["CPUExecutionProvider"])


_P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
_P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
_P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])


def dnsmos(x: np.ndarray) -> dict:
    audio = np.asarray(x, dtype=np.float32)
    len_samples = int(DNSMOS_LEN * SR)
    while audio.shape[-1] < len_samples:
        audio = np.concatenate([audio, audio], axis=-1)
    num_hops = int(np.floor(audio.shape[-1] / SR) - DNSMOS_LEN) + 1
    sess = _dnsmos_session()
    rows = []
    for idx in range(num_hops):
        seg = audio[int(idx * SR): int((idx + DNSMOS_LEN) * SR)]
        if seg.shape[-1] < len_samples:
            continue
        out = sess.run(None, {"input_1": seg.reshape(1, -1).astype("float32")})[0].astype("float64")
        rows.append(out[0])
    r = np.mean(np.stack([[_P_SIG(a), _P_BAK(b), _P_OVR(c)] for a, b, c in rows]), axis=0)
    return {"sig": float(r[0]), "bak": float(r[1]), "ovrl": float(r[2])}


@lru_cache(maxsize=4)
def _srmr_filters(fs: int = SR, n_filters: int = 23, low_freq: float = 125.0, min_cf: float = 4.0,
                  max_cf: float = 128.0):
    from gammatone.filters import centre_freqs, make_erb_filters

    cfs = centre_freqs(fs, n_filters, low_freq)
    erb = make_erb_filters(fs, cfs)
    erbs = np.flipud((cfs / 9.26449) + 24.7)  # Glasberg & Moore ERB of each band, low to high
    spacing = (max_cf / min_cf) ** (1.0 / 7)
    mcf = min_cf * spacing ** np.arange(8)
    mod = []
    for w0 in 2 * np.pi * mcf / fs:
        t = np.tan(w0 / 2)
        b0 = t / 2  # Q = 2
        mod.append((np.array([b0, 0.0, -b0]), np.array([1 + b0 + t * t, 2 * t * t - 2, 1 - b0 + t * t])))
    ll = mcf - (np.tan(2 * np.pi * mcf / fs / 2) / 2 * fs / (2 * np.pi))
    return erb, erbs, mod, ll


def srmr(x: np.ndarray, fs: int = SR) -> float:
    """SRMR (norm off, 23 gammatone bands, 8 modulation bands 4-128 Hz), the torchmetrics/SRMRpy
    algorithm computed one band at a time. Same numbers as torchmetrics (tested), but a 60 s window
    needs tens of MB instead of about 15 GB: torchmetrics keeps all 23 x 8 band signals at once."""
    from scipy import fft as sfft
    from scipy import signal as ss

    x = np.asarray(x, dtype=np.float64)
    m = np.max(np.abs(x))
    if m > 1:
        x = x / m
    erb, erbs, mod, cutoffs = _srmr_filters(fs)
    n = len(x)
    n_fft = int(np.ceil(n / 16) * 16)
    h = np.zeros(n_fft)
    if n_fft % 2 == 0:
        h[0] = h[n_fft // 2] = 1
        h[1: n_fft // 2] = 2
    else:
        h[0] = 1
        h[1: (n_fft + 1) // 2] = 2
    wl, wi = int(np.ceil(0.256 * fs)), int(np.ceil(0.064 * fs))
    n_frames = int(1 + (n - wl) // wi)
    win = np.hamming(wl + 1)[:-1]
    avg = np.zeros((erb.shape[0], 8))
    for c in range(erb.shape[0]):
        A0, A11, A12, A13, A14, A2, B0, B1, B2, gain = erb[c]
        y = x
        for a1 in (A11, A12, A13, A14):
            y = np.clip(ss.lfilter([A0, a1, A2], [B0, B1, B2], y), -1.0, 1.0)  # torchaudio clamps here
        y = y / gain
        env = np.abs(sfft.ifft(sfft.fft(y, n=n_fft) * h)[:n])
        for k, (bk, ak) in enumerate(mod):
            mo = ss.lfilter(bk, ak, env)
            fr = np.lib.stride_tricks.sliding_window_view(mo, wl)[::wi][:n_frames]
            avg[c, k] = np.mean(np.sum((fr * win) ** 2, axis=1))
    total = avg.sum()
    perc = avg.sum(axis=1) * 100 / total
    k90 = int(np.argmax(np.cumsum(perc[::-1]) > 90))
    bw = erbs[k90]
    if cutoffs[4] <= bw < cutoffs[5]:
        kstar = 5
    elif cutoffs[5] <= bw < cutoffs[6]:
        kstar = 6
    elif cutoffs[6] <= bw < cutoffs[7]:
        kstar = 7
    elif cutoffs[7] <= bw:
        kstar = 8
    else:
        raise ValueError("SRMR: band edge below the modulation cutoffs")
    return float(avg[:, :4].sum() / avg[:, 4:kstar].sum())


def prep(x: np.ndarray) -> np.ndarray:
    return (x / (np.max(np.abs(x)) + 1e-9) * 0.5).astype(np.float32)


def score(x16: np.ndarray, with_srmr: bool = True) -> dict:
    """All no-reference scores of one 16 kHz mono window."""
    y = prep(x16)
    d = dnsmos(y)
    if with_srmr:
        d["srmr"] = srmr(y)
    return {k: round(v, 3) for k, v in d.items()}
