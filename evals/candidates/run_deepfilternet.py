"""DeepFilterNet 3 (MIT/Apache-2.0): noise suppression at 48 kHz."""
from _common import run


def setup():
    from df.enhance import init_df
    model, st, _ = init_df()
    return model, st


def one(state, src, dst):
    from df.enhance import enhance, load_audio, save_audio
    model, st = state
    audio, _ = load_audio(src, sr=st.sr())
    save_audio(dst, enhance(model, st, audio), st.sr())


run(one, setup)
