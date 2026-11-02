"""make-normal-voice engines inside the make-normal-voice environment:
sidon | master | sidon-onnx-int8 | sidon-onnx-fp16 | sidon-declip | sidon-notch (hypothesis branches)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "skills" / "make-normal-voice" / "src"))
from _common import run  # noqa: E402

MODE = sys.argv.pop(3) if len(sys.argv) > 3 else "sidon"


def setup():
    if MODE == "master":
        return None
    from make_normal_voice.engines import sidon
    if MODE in ("sidon-declip", "sidon-notch"):
        return sidon.Sidon("cpu")
    if MODE.startswith("sidon-onnx"):
        from sidon_onnx import load
        return sidon.Sidon(models=load(MODE.rsplit("-", 1)[1]))
    return sidon.Sidon("cpu")


def one(eng, src, dst):
    import shutil
    import tempfile

    from make_normal_voice import master
    from make_normal_voice.engines import sidon
    from make_normal_voice.stages import source_len48
    if MODE == "master":
        master.master(src, dst)
        return
    if MODE == "sidon-declip":
        from make_normal_voice import ffmpeg
        x = ffmpeg.decode(src, sr=16000, channels=1, af="adeclip,highpass=f=50")[0][:, 0]
        x = (0.9 * x / (abs(x).max() + 1e-9)).astype("float32")
    elif MODE == "sidon-notch":
        from make_normal_voice import analyze, ffmpeg, vad
        from make_normal_voice.audio import load_mono
        x16 = load_mono(src)
        hum = analyze._hum(x16, vad.pauses(vad.speech_segments(x16), len(x16) / 16000))
        f0 = hum["freq"] or 50
        notch = ",".join(f"bandreject=f={f0 * h}:width_type=q:w=30" for h in range(1, 5))
        x = ffmpeg.decode(src, sr=16000, channels=1, af=notch + ",highpass=f=50")[0][:, 0]
        x = (0.9 * x / (abs(x).max() + 1e-9)).astype("float32")
    else:
        x = sidon.prepare_input(src)
    tmp = Path(tempfile.mkdtemp()) / "s.wav"
    eng.restore(x, tmp, target=source_len48(Path(src)))
    shutil.move(str(tmp), dst)


run(one, setup)
