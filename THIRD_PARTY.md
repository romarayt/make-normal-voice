# Third-party components

make-normal-voice's own code is MIT. Nothing below is vendored into this repository: models are
downloaded by `make-normal-voice install` from the pinned revision listed here and checked against the
sha256; Python packages are installed by uv from `skills/make-normal-voice/uv.lock`.

## Models (downloaded on demand)

| Component | Used for | License | Source (pinned) | sha256 |
|---|---|---|---|---|
| Sidon v0.1 feature extractor, CPU (`feature_extractor_cpu.pt`, 795 MB) | speech restoration, default engine | MIT | [sarulab-speech/sidon-v0.1](https://huggingface.co/sarulab-speech/sidon-v0.1) @ `b3b02d8b` | `fd9abc906a9048b3c047bd3d246a25f6485d09c2d6bb85e098f527f844efc019` |
| Sidon v0.1 decoder, CPU (`decoder_cpu.pt`, 210 MB) | speech restoration, default engine | MIT | same | `34dcc80fab75bd1336369ba5b2063f6da475887aa824db94b495c4059c5fdea4` |
| Sidon v0.1 feature extractor, CUDA (optional, 795 MB) | restoration on NVIDIA GPUs | MIT | same | `c3739c332da4ce00cc9019a328bcb19ee8f863ac5eb0b52386672772aab836ce` |
| Sidon v0.1 decoder, CUDA (optional, 210 MB) | restoration on NVIDIA GPUs | MIT | same | `a789c71f54ca7db4b28becdc87c5e8b9260cb568cca462d502c54900db74a1cc` |
| DNSMOS P.835 `sig_bak_ovr.onnx` (1.2 MB) | no-reference quality score | CC BY 4.0 (repository license) | [microsoft/DNS-Challenge](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS) @ `82f1b17e` | `269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd` |
| WeSpeaker ECAPA-TDNN512 VoxCeleb (`voxceleb_ECAPA512.onnx`, 25 MB) | same-voice check | Apache-2.0 | [Wespeaker/wespeaker-voxceleb-ecapa-tdnn512](https://huggingface.co/Wespeaker/wespeaker-voxceleb-ecapa-tdnn512) @ `89426e11` | `f901ae3571912bad4e11e157744cb8b7d77a0b46923b83c410bdbb6c77d69736` |
| faster-whisper small (CTranslate2, 486 MB) | word check, end-of-speech for trims | MIT | [Systran/faster-whisper-small](https://huggingface.co/Systran/faster-whisper-small) @ `536b0662` | `model.bin` `3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671` (other files in `install.py`) |
| faster-whisper large-v3-turbo (optional, 1.6 GB) | more accurate word check | MIT | [deepdml/faster-whisper-large-v3-turbo-ct2](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2) @ `4df90f75` | `model.bin` `e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da` |

Sidon (Nakata, Saito et al., [arXiv:2509.17052](https://arxiv.org/abs/2509.17052)) is a fine-tune of
[facebook/w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0) (MIT) trained on FLEURS-R and
LibriTTS-R. Its feature front end is `SeamlessM4TFeatureExtractor` with the parameters of
w2v-bert-2.0's `preprocessor_config.json`, constructed in code (no download).

Silero VAD (MIT) is used through the ONNX file bundled in the `faster-whisper` wheel.

## Python packages (installed from the lock file)

numpy, scipy (BSD-3-Clause) · soundfile (BSD-3-Clause; bundles libsndfile, LGPL-2.1) ·
soxr (LGPL-2.1, python-soxr) · platformdirs (MIT) · torch (BSD-3-Clause) ·
transformers (Apache-2.0) · gammatone (BSD-3-Clause; gammatone filter design for SRMR) ·
onnxruntime (MIT) · faster-whisper (MIT) with CTranslate2 (MIT) · matplotlib (PSF-based) ·
dev/eval only: pytest (MIT), ruff (MIT), pesq (MIT), pystoi (MIT), torchmetrics and torchaudio
(Apache-2.0 / BSD-2-Clause; only to check that our SRMR and DNSMOS give the same numbers).

The SRMR and DNSMOS scoring code follows the SRMRpy / torchmetrics (Apache-2.0) and Microsoft
DNS-Challenge reference algorithms; it is reimplemented, not copied.

## External tools

ffmpeg / ffprobe (LGPL-2.1 or GPL depending on the build), installed by the user's package manager.

## Evaluation data (fetched by `evals/fetch_data.py`, never committed)

| Data | License | Source |
|---|---|---|
| LibriTTS-R test-clean | CC BY 4.0 | <https://www.openslr.org/141/> |
| FLEURS ru_ru test | CC BY 4.0 | <https://huggingface.co/datasets/google/fleurs> |
| RIRS_NOISES (real RIRs and isotropic noises) | Apache-2.0 | <https://www.openslr.org/28/> |

The only audio in the repository is `tests/fixtures/*.flac`, cut from LibriTTS-R test-clean
(CC BY 4.0); see `tests/fixtures/ATTRIBUTION.md`.

## Benchmark candidates (not part of the skill)

Installed separately for `evals/bench.py` only: DeepFilterNet (MIT/Apache-2.0),
ClearerVoice-Studio MossFormer2_SE_48K (Apache-2.0), VoiceFixer (MIT), Resemble Enhance (MIT),
soniqo/Sidon-ONNX (MIT).
