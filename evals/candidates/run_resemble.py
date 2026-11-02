"""Resemble Enhance (MIT): denoise + latent-diffusion enhancement, 44.1 kHz output.

Its own downloader clones the weights with git-lfs; the benchmark places the pinned checkpoint
directly (see evals/README.md) and skips that step."""
from _common import run


def setup():
    import resemble_enhance.enhancer.download as dl
    import resemble_enhance.enhancer.inference as inf

    inf.download = lambda: dl.REPO_DIR / "enhancer_stage2"
    return None


def one(_, src, dst):
    import soundfile as sf
    import torch
    from resemble_enhance.enhancer.inference import enhance
    x, sr = sf.read(src, dtype="float32", always_2d=True)
    wav = torch.from_numpy(x[:, 0])
    y, sr2 = enhance(wav, sr, "cpu", nfe=32, solver="midpoint", lambd=0.9, tau=0.5)
    sf.write(dst, y.cpu().numpy(), sr2, subtype="FLOAT")


run(one, setup)
