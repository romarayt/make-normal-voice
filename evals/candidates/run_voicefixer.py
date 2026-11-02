"""VoiceFixer (MIT): general speech restoration, 44.1 kHz output."""
from _common import run


def setup():
    from voicefixer import VoiceFixer
    return VoiceFixer()


def one(vf, src, dst):
    vf.restore(input=src, output=dst, cuda=False, mode=0)


run(one, setup)
