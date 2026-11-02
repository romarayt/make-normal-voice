# Make Normal Voice

**An Agent Skill that turns any speech recording into an edit-ready voice track that sounds
close-miked, like a podcast microphone. Locally, with measurements instead of "sounds good".**

[Русская версия](README.ru.md) · by Roma Rayt: [Telegram channel](https://t.me/romarayt) · [YouTube channel «Нейропросвещение»](https://www.youtube.com/@neuropros)

Camera audio with room echo, a laptop's built-in mic, a screen recording, a noisy lavalier, several
devices of one session: the skill analyses each track, explains the route in one line, restores the
voice with [Sidon](https://huggingface.co/sarulab-speech/sidon-v0.1), mixes it without a compressor
to −16 LUFS, then **verifies** the result (quality, words, voice, 1 ms sync, loudness, seams) and
fixes what fails. Your audio never leaves your machine.

It works in **Claude Code** (plugin or skill folder) and **Codex** (skill folder), and as a plain CLI.

## Example

An open clip (LibriTTS-R, CC BY 4.0) played in a real small room through a far mic, with room noise
and at -38 LUFS, like camera audio (details in [docs/samples](docs/samples/ATTRIBUTION.md)):

| | DNSMOS OVRL | DNSMOS BAK | loudness |
|---|---|---|---|
| before ([before.mp3](docs/samples/before.mp3)) | 3.11 | 4.00 | -38 LUFS |
| after ([after.mp3](docs/samples/after.mp3)) | 3.49 | 4.17 | -16 LUFS, -1.5 dBTP |

Words unchanged (0 of 45 confident words), same voice (similarity 0.91), length exact. The two
probes are the same 30 s at equal loudness. Spectrograms:
[before](docs/samples/spectrogram_before.png) · [after](docs/samples/spectrogram_after.png).


## Install

Requirements: macOS or Linux, [ffmpeg](https://ffmpeg.org) and [uv](https://docs.astral.sh/uv/).
About 1 GB of Python packages and 1.5 GB of models are downloaded on first use (asked once).

**Claude Code, as a plugin** (inside Claude Code):

```
/plugin marketplace add romarayt/make-normal-voice
/plugin install make-normal-voice@make-normal-voice
```

or from a shell: `claude plugin marketplace add romarayt/make-normal-voice && claude plugin install make-normal-voice@make-normal-voice`.

**Claude Code or Codex, as a skill folder:**

```bash
git clone https://github.com/romarayt/make-normal-voice ~/.local/share/make-normal-voice-src
ln -s ~/.local/share/make-normal-voice-src/skills/make-normal-voice ~/.claude/skills/make-normal-voice   # Claude Code
ln -s ~/.local/share/make-normal-voice-src/skills/make-normal-voice ~/.agents/skills/make-normal-voice   # Codex
```

Then ask the agent: *"clean up the audio of interview.mp4"*, *"remove the echo from camera.wav"*,
*"почисти звук"*, *"убери эхо"*. Or run it yourself:

```bash
MNV=~/.local/share/make-normal-voice-src/skills/make-normal-voice/scripts/make-normal-voice
$MNV doctor            # what is there, what is missing
$MNV install           # downloads the models (pinned revisions, sha256-checked, resumable)
$MNV plan camera.wav   # camera.wav: echo (SRMR 5.9, OVRL 2.53), quiet (-40 LUFS) → Sidon → mix
$MNV process camera.wav --out cleaned/
```

## What you get

For every track, in the output folder:

- `<name>.clean.wav`: 48 kHz, 24-bit, dual-mono stereo, −16 LUFS (configurable), true peak ≤ −1 dBTP,
  **same length and start as the source** (sample-exact), unless you asked for a trim;
- `report.md` in plain language (`--lang ru` for Russian): what the recording was, what was done,
  before → after numbers, every gate that failed or could not be verified, what was cut;
- `metrics.json` with everything measured;
- `ab_before_30s.wav` / `ab_after_30s.wav`: the same 30 s at equal loudness, to decide by ear;
- `spectrogram_before.png` / `spectrogram_after.png`;
- `probe/` for long recordings (the hardest 2.5 minutes, processed first), and
  `junction_probe.wav` when several sources were glued.

Several files of one session (camera + screen recording + lavalier):
`process cam.mp4 screen.mov lav.wav --sync-to cam.mp4 --out session/`. The skill checks by the speech
envelope that they belong together (own session ≈ 0.98, a foreign recording 0.2-0.3), aligns them to
1 ms regardless of timecodes and clocks, removes clock drift, takes the best source and fills its gaps
from the others, matches timbre, and puts the seams in pauses.

## How it works

```
analyze ─► route ─► probe (hardest 150 s) ─► restore ─► level parts ─► match timbre ─► mix ─► verify ─► fix (≤3 rounds)
```

- **Restore: Sidon** (University of Tokyo, MIT). It does not subtract echo; it rebuilds the voice
  from w2v-BERT 2.0 speech features as if recorded by a close mic, at 48 kHz. make-normal-voice runs it
  with sample-exact chunking: the reference recipe loses 26 samples per 96 s chunk (5.6 ppm, about
  20 ms per hour, enough to drift off the picture) and adds a 7.4 ms delay; both are fixed here
  (see [DECISIONS.md](DECISIONS.md), D5-D7).
- **Mix**: high-pass 70 Hz, light de-esser, one linear gain to the target measured on the final
  stereo, soft limiter with latency compensation. No compressor: it brings the room back.
- **Verify**: every gate below runs on the delivered file. A failed gate triggers a targeted fix;
  after three rounds the best round is delivered and the report says what still fails.

| Gate | Passes when |
|---|---|
| podcast-mic target | median over speech windows: DNSMOS OVRL ≥ 3.3, BAK ≥ 4.0 (SRMR is reported: it depends on the voice, see below) |
| not worse than input | OVRL not lower on any window; SRMR not lower (for already dry inputs: ≥ 9) |
| words unchanged | confidently recognised input words survive, beyond the recogniser's own noise |
| same voice | speaker-embedding similarity input vs output ≥ 0.65 |
| sync and length | median offset ≤ 1 ms at 9 points; length exact to the sample |
| loudness | −16 ± 0.5 LUFS, parts within ±1 LU, true peak ≤ −1 dBTP |
| timbre (with `--reference`) | third-octave speech spectrum within ±1 dB of the reference |
| seams, lost speech | no sample-level click and no background jump at chunk/part seams; no input speech silent in the output |

Each gate is tested against a deliberately broken output (added echo, 50 ms shift, +3 LU, one
replaced word, another person's voice, a click at a seam, a background jump): it must fail.

<!-- BENCH-START -->
## Benchmark

Open speech only: 6 English references (LibriTTS-R test-clean, CC BY 4.0, not in Sidon's training
data) and 6 Russian (LibriVox readings, public domain), each degraded into 12 classes with real room
responses and noises from OpenSLR 28: 144 inputs. Eight engines (plus two hypothesis branches) were run on all of them, one at a
time on CPU. Medians per class; OVRL/BAK = DNSMOS (higher is better), PESQ/ESTOI against the clean
original, WER against what the same recogniser hears in the clean original, "voice" = speaker
similarity to the clean original. Full table: [evals/results/bench.md](evals/results/bench.md).

| Class | Input OVRL | Sidon OVRL / BAK | Sidon PESQ / ESTOI | Sidon WER / voice | Best other by OVRL (its WER) |
|---|---|---|---|---|---|
| clean (repeatability) | 3.42 | 3.45 / 4.18 | 2.70 / 0.89 | 0.01 / 0.91 | deepfilternet 3.45 (WER 0.00) |
| far mic, room echo | 1.84 | 3.46 / 4.21 | 2.00 / 0.79 | 0.06 / 0.86 | resemble-enhance 3.32 (WER 0.84) |
| far mic, echo + room noise | 1.62 | 3.50 / 4.21 | 1.98 / 0.77 | 0.08 / 0.85 | resemble-enhance 3.38 (WER 0.96) |
| laptop-like (small room, band-limited, AAC) | 2.42 | 3.49 / 4.20 | 2.07 / 0.74 | 0.06 / 0.85 | resemble-enhance 3.39 (WER 0.71) |
| dry voice, loud noise (5 dB SNR) | 2.37 | 3.51 / 4.21 | 2.05 / 0.84 | 0.08 / 0.82 | resemble-enhance 3.41 (WER 0.17) |
| clipping | 3.23 | 3.45 / 4.20 | 2.62 / 0.87 | 0.06 / 0.89 | sidon-declip 3.47 (WER 0.03) |
| phone (300-3400 Hz, 8 kHz µ-law) | 3.32 | 3.46 / 4.19 | 2.34 / 0.89 | 0.03 / 0.77 | resemble-enhance 3.41 (WER 0.08) |
| MP3 24 kbps | 3.37 | 3.44 / 4.19 | 2.43 / 0.88 | 0.01 / 0.83 | resemble-enhance 3.41 (WER 0.05) |
| 50/60 Hz hum | 3.28 | 3.48 / 4.21 | 2.63 / 0.87 | 0.02 / 0.91 | sidon-notch 3.45 (WER 0.01) |
| very quiet (-40 LUFS, 16-bit) | 3.37 | 3.43 / 4.19 | 2.70 / 0.89 | 0.01 / 0.91 | clearvoice 3.43 (WER 0.00) |
| digital dropouts | 3.26 | 3.36 / 4.17 | 1.91 / 0.84 | 0.03 / 0.89 | deepfilternet 3.33 (WER 0.04) |
| music under the voice | 2.71 | 3.49 / 4.21 | 2.31 / 0.85 | 0.00 / 0.90 | resemble-enhance 3.40 (WER 0.02) |

What this decides:

- **Sidon is the default for everything that has a room in it** (far mic, echo + noise, laptop): DNSMOS
  OVRL ~3.5 from 1.6-2.4 with WER 0.06-0.08, ahead of every alternative. Resemble Enhance also *sounds*
  clean to DNSMOS (3.3-3.4) but says something else: WER 0.71-0.96, voice similarity 0.55-0.63, and it
  is slower than real time on CPU. VoiceFixer reaches 3.2 with WER 0.19-0.24 and a changed voice
  (0.68-0.70). DeepFilterNet and ClearerVoice remove noise, not the room (2.3-3.0).
- **A clean input is left alone** (route "mix only"). Sidon does not make it worse by DNSMOS (3.45 vs
  3.42), but it regenerates the waveform (PESQ 4.6 → 2.7), so the router only mixes clean close-mic
  tracks unless they must be glued to restored parts.
- **Noise without a room**: Sidon is cleaner by DNSMOS (3.51 vs 3.34), while DeepFilterNet/ClearerVoice
  keep the original waveform more faithfully (PESQ 2.6 vs 2.1). make-normal-voice aims at the podcast-mic
  sound, so it uses Sidon; a classic denoiser is not built in.
- **Clipping, phone band, low-bit MP3, hum, very quiet**: Sidon alone already reaches the target. The
  hypothesis branches (`adeclip` or a hum notch before Sidon) are in the table as `sidon-declip` /
  `sidon-notch`; neither beat Sidon alone (declip: OVRL 3.47 vs 3.45 but PESQ 2.54 vs 2.62 and 5x slower; notch: 3.45 vs 3.48), so both stay off (`--experimental declip,hum_notch` turns them on).
- **The Sidon ONNX port** (soniqo/Sidon-ONNX, int8 and fp16) sounds the same but is about 2.5x slower
  on CPU (real-time factor 0.37-0.40 vs 0.15); the TorchScript weights stay the default.
- **Music under the voice** is removed rather than kept: DNSMOS rates the result as clean speech, which
  is exactly why this case is a warning, not a feature.
- **Timing**: against the clean originals Sidon's output sits at a median +0.1…+0.25 ms in every class
  (+1.0 ms with echo and noise); an hour of speech stays within ±0.26 ms (`evals/results/sidon_drift.json`).
<!-- BENCH-END -->

## Honest limits

- **Proven vs experimental.** The default route (Sidon → mix) is proven on real recordings
  (camera with room echo, laptop mic during screen recording, lavalier) and on the benchmark above.
  Branches marked experimental are off by default and say so in the plan.
- **It rebuilds the voice.** Sidon does not subtract the room, it re-synthesises speech from speech
  features. The voice stays recognisably the same (speaker similarity 0.87-0.95 in practice), but the
  timbre can shift slightly; `--reference` matches it to a track you already like.
- **Measured, not listened to.** The gates are proxies. The report says so and gives you 30 s
  before/after probes at equal loudness. Your ear has the last word.
- **Speed.** CPU only was measured: about 5x faster than real time on an Apple M4 Max (a 20-minute
  track in about 4 minutes, an hour in about 13), plus 1-3 minutes of checks. GPU (CUDA) weights are
  available but untested in v0.2.0.
- **Memory.** About 5 GB at the peak (restoration in 30 s chunks, decoder in 3 s windows). The tool
  checks free memory before and during the run and stops cleanly instead of making the computer swap.
  Run one job at a time.
- **Disk.** Python environment about 1.0 GB, models 1.5 GB (+1 GB optional CUDA weights, +1.6 GB
  optional large recogniser); the restoration cache keeps about 0.7 GB per hour of audio so re-runs
  take minutes; about 4 GB of working space per hour of audio during a run.
- **Not handled in v0.2.0.** Music under the voice (warned, may be removed or garbled), two people
  talking at once (not detected), dropouts without a second source (reported, not patched), Windows
  (launcher included, untested).
- **Sync against very echoey inputs** cannot be measured to 1 ms by envelope (the echo smears the
  input); the gate then says "not verified". The engine's own timing is proven on synthetic echo and
  on an hour of speech (`evals/results/sidon_drift.json`).
- **Word check.** Uses Whisper small locally; its own noise is subtracted, but it is weaker on
  non-English speech. `--asr-model turbo` (1.6 GB) is more accurate.

## Development

```bash
uv run --project skills/make-normal-voice --group dev ruff check skills/make-normal-voice/src tests evals
uv run --project skills/make-normal-voice --group dev pytest -q tests
```

CI runs lint and the fast tests (synthetic signals and short open clips) on Linux and macOS; the
tests that need the 1 GB Sidon weights run in a manual workflow. The benchmark is in
[evals/](evals/README.md).

## Author

Roma Rayt: [Telegram channel](https://t.me/romarayt) and YouTube channel [«Нейропросвещение»](https://www.youtube.com/@neuropros), both in Russian.

## License

MIT for this code. Models and libraries are downloaded at install time under their own licenses:
see [THIRD_PARTY.md](THIRD_PARTY.md). Test fixtures are cut from LibriTTS-R (CC BY 4.0).
