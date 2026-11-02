"""ClearerVoice-Studio MossFormer2_SE_48K (Apache-2.0): speech enhancement at 48 kHz."""
from _common import run


def setup():
    from clearvoice import ClearVoice
    return ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])


def one(cv, src, dst):
    out = cv(input_path=src, online_write=False)
    cv.write(out, output_path=dst)


run(one, setup)
