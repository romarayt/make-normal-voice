# Evaluation

Nothing here is needed to use the skill. It is how the routes and thresholds were chosen.

```bash
export VR_EVAL_DATA=~/.cache/voice-rescue-evals        # downloads go here, never into the repo
export VR_CANDIDATES=~/.cache/voice-rescue-candidates  # one venv per competing engine
PY="uv run --project skills/voice-rescue --group dev --group eval python"
$PY evals/fetch_data.py                 # LibriTTS-R test-clean (EN), Russian LibriSpeech (RU), OpenSLR 28 RIRs/noises
$PY evals/bench.py prepare              # clean statistics + 12 degraded classes per reference
$PY evals/bench.py run sidon master-only sidon-onnx-int8 sidon-onnx-fp16 deepfilternet clearvoice voicefixer resemble-enhance
$PY evals/bench.py score
$PY evals/bench.py report               # -> evals/results/bench.md, bench.json
$PY evals/sidon_drift.py 60             # one hour through Sidon: offset at 20 points, vs the naive gluing
```

Heavy candidates use all CPU cores: run them one at a time.

Competing engines are installed outside voice-rescue's environment (each has conflicting pins):

| Candidate | venv | Install |
|---|---|---|
| DeepFilterNet 3 | Python 3.11 | `torch==2.1.2 torchaudio==2.1.2 "numpy<2" deepfilternet` |
| ClearerVoice MossFormer2_SE_48K | Python 3.11 | `clearvoice` |
| VoiceFixer | Python 3.10 | `voicefixer "numpy<2"` (checkpoints from zenodo, ~620 MB) |
| Resemble Enhance | Python 3.11 | `torch==2.2.2 torchaudio==2.2.2` + deps, then `resemble-enhance --no-deps`; it imports `deepspeed` even for inference (the benchmark uses an empty stub module) and fetches its weights with git-lfs (the benchmark places the pinned checkpoint, sha256 `f9d035f3…`, directly) |
| Sidon ONNX (soniqo) | voice-rescue's | files from `soniqo/Sidon-ONNX` into `$VR_EVAL_DATA/sidon-onnx/{int8,fp16}` |

Scoring: outputs are aligned to the clean reference (cross-correlation, ±150 ms), then PESQ (wide
band) and ESTOI against it, DNSMOS and SRMR without it, word error rate against what the same
recogniser hears in the clean original, speaker similarity to the clean original, and real-time
factor. SI-SDR is not used: Sidon regenerates the waveform.

Classes: `clean` (repeatability), `far_echo` (real RIR, T60 ≥ 0.45 s), `far_echo_noise` (+ real
room noise at 12 dB SNR), `laptop` (small-room RIR, 150 Hz-12 kHz, pink noise 22 dB, AAC 96k),
`noise_dry` (real noise, 5 dB SNR), `clipping`, `phone` (300-3400 Hz, 8 kHz µ-law), `codec_lowbit`
(MP3 24 kbps at 22 kHz), `hum` (50/60 Hz + harmonics at -20 dB), `quiet` (-40 LUFS, 16-bit, -80 dBFS
hiss), `dropouts` (six 50-150 ms digital gaps), `music` (synthetic chords and hats at -12 dB).
