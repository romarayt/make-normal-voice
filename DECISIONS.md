# Decisions

Every non-obvious choice, why it was made and on which numbers. Numbers from the open benchmark
are in `evals/results/`; observations on private recordings are described without their numbers
(those recordings and their measurements stay off this repository).

## Name and packaging

- **D1. Name `make-normal-voice`.** The name says what the skill does, for people who never heard of
  speech restoration (the author's request; 0.1.0 was published as `voice-rescue`, and GitHub
  redirects the old repository address). No repository or package of that name exists.
- **D2. One repository = Claude Code plugin = marketplace; the skill is `skills/make-normal-voice/`.**
  The skill folder is self-contained (SKILL.md, references, launcher, the Python project and its
  lock file), so it can also be copied or symlinked into `~/.claude/skills/` or `~/.agents/skills/`
  (Codex). `name` equals the folder name as the Agent Skills spec requires. Validated with
  `claude plugin validate`.
- **D3. A shell launcher + uv, the environment in the per-user data folder.** The skill runs
  `scripts/make-normal-voice`, which calls `uv run --frozen` with `UV_PROJECT_ENVIRONMENT` set to
  `<data folder>/env` (platformdirs location; `MAKE_NORMAL_VOICE_HOME` overrides). Reasons: a broken
  system Python (seen: Homebrew Python without `pyexpat`, so pip and uv refused to install) does
  not matter; nothing lives in `/tmp` (macOS wiped a venv there after 3 days); versions are pinned
  by `uv.lock`. Linux pulls CPU wheels of torch (a CUDA build is 2+ GB and most machines lack a GPU).
- **D4. Models are downloaded by `install`, never vendored.** Pinned revisions, sha256 checked,
  resumable with HTTP Range, progress and timeouts; asks once (`--yes` for unattended runs);
  `--from <folder>` imports existing files after the checksum. Model files are data: they are
  loaded by torch/onnxruntime, never imported as code.

## The Sidon engine

- **D5. Root cause of the 5.6 ppm time squeeze.** Measured directly: the TorchScript decoder
  returns `frames × 960 − 26` samples for every call, aligned at the start (sub-decodes of the same
  features match the full decode at offset 0 to 1e-7, and lose the last 26 samples). The reference
  demo decodes 96 s chunks and concatenates them, so every chunk is 26 samples short:
  26 / 4 608 000 = 5.64 ppm, ≈20 ms per hour, exactly the drift seen in practice. The demo then
  pads the end to the target length, which hides the loss at the end but not along the way.
- **D6. Fix: context chunks with an exact crop.** Each chunk is decoded with 150 frames (3 s)
  of real audio on each side through both the feature extractor and the decoder, then cropped to
  exactly `frames × 960` samples; the 10 ms seam crossfade blends two decodes of the same samples.
  Chunk boundaries sit on the 20 ms frame grid. Tests: a stand-in decoder with the same two
  properties gives a sample-exact hour (37 seams); with the real weights, 10 s chunks match a single
  chunk and show no click at the seams.
- **D7. Constant decoder delay 354 samples (7.375 ms) removed.** Measured by 1 ms envelope
  cross-correlation against the input on a laptop-mic recording and a lavalier recording (median of
  25 windows each, 7.35 and 7.38 ms, scatter ±0.5 ms). On an echoey camera recording single windows
  scatter 1-8 ms because the reverb smears the input envelope; the median agrees within the scatter.
  The hour test on open speech is in `evals/results/sidon_drift.json`.
- **D8. Input level.** As in the reference demo: 50 Hz high-pass, 16 kHz, peak-normalised to 0.9.
  Sidon's feature normalisation makes the output level largely independent of the input level: two
  parts of a recording 6.5 LU apart came out within 0.1 LU, so "level the parts" now re-measures
  after restoration and applies only the residual gain.
- **D9. TorchScript CPU weights by default; ONNX port not by default.** See the benchmark for the
  soniqo/Sidon-ONNX int8/fp16 variants (same decoder, same 26-sample tail).

- **D9a. Memory: 30 s chunks and a windowed decoder.** The first full runs with the demo's 96 s
  chunks used 12-16 GB and pushed a 36 GB machine into swap until it froze. Measured on CPU: the
  feature extractor grows with the square of the chunk (+0.5 GB at 20 s, +1.8 GB at 40 s, +4.1 GB at
  60 s, about +12 GB at 102 s with context), the decoder linearly (about 0.2 GB per second of audio,
  because it keeps 48 kHz activations). Now features are computed on 30 s chunks (+3 s context each
  side) and the decoder runs on 3 s windows with 0.32 s of feature context, cropped exactly; windowed
  decoding equals one long decoder call to 1.5e-7. A whole Sidon run peaks at about 4.4 GB. Quality on
  the same echoey camera excerpt: OVRL 3.43 / 3.35 with 30 s chunks against 3.43 / 3.31 with 96 s
  (Sidon was trained on 5-20 s utterances, so shorter chunks are not a handicap). Before starting and
  before every chunk the tool checks free memory; it shortens the chunk to 20 s or stops with a clear
  message instead of letting the machine swap. "Free" is the kernel's own estimate on macOS
  (`memory_pressure`, which counts reclaimable file cache; raw `vm_stat` free+inactive pages read about
  half of it) and `MemAvailable` on Linux.

## Mix

- **D10. The proven chain is kept as is**: high-pass 70 Hz, light de-esser, one linear gain to
  −16 LUFS measured on the final dual-mono stereo, `alimiter` with `latency=1`, 48 kHz/24-bit. No
  compressor (it brought the room back; SRMR fell below the untreated input), no stereo widening.
  The limiter is lowered automatically if true peak exceeds −1 dBTP.
- **D11. Timbre matching with a linear-phase FIR in numpy** instead of ffmpeg `firequalizer`, whose
  `delay` shifts the audio even with `zero_phase`. The FIR delay `(N−1)/2` is removed exactly; a
  unit test checks an impulse stays on its sample.

## Verification gates

- **D12. DNSMOS and SRMR are our own implementations of the reference algorithms.** DNSMOS: so the
  model file comes from our pinned, checksummed download (torchmetrics hard-codes `~/.torchmetrics`);
  identical to torchmetrics to 4 decimals on 4 test windows. SRMR: torchmetrics keeps all 23
  gammatone × 8 modulation band signals in float64 at once, 3.3 GB for a 10 s window and about
  15 GB for the 60 s windows we score; together with a running restoration that froze the machine.
  Ours computes the same algorithm one band at a time: equal to torchmetrics within 1e-4 on five
  windows, a 60 s window in about 1 s and tens of MB. torchmetrics and torchaudio are no longer
  dependencies (only the eval group uses them, to check equality). Averaging SRMR over 10 s pieces was
  rejected: it deviates from the whole-window value by up to 13%.
- **D13. Speaker model: WeSpeaker ECAPA-TDNN512 (Apache-2.0, ONNX, 25 MB)** instead of SpeechBrain:
  no extra framework, onnxruntime is already needed for DNSMOS. Kaldi-style fbank is computed in
  numpy. Same person in vs out: 0.87-0.95; another person: far below 0.55 (benchmark).
- **D14. Word gate on *confidently recognised* input words, not plain WER.** On an echoey camera
  recording plain WER flagged "changed words" because the recogniser could not make out the echoey
  input, while the restored output was understood better. The fix loop then put the unrestored
  input back on those stretches and the echo returned. Now only input words with recogniser
  probability ≥ 0.8 and ≥ 3 letters count; a word whose letters mostly match (changed ending) is not
  lost; the recogniser's own noise (beam 5 vs greedy on the input) is subtracted; margin 5 points.
  The broken-input test replaces one word and must fail. Plain WER is still reported.
- **D15. The fix loop keeps the best round.** Quality failures weigh most, so a fix can never
  trade the echo back for another gate; a round that makes things worse stops the loop and the
  better earlier round is delivered.
- **D15a. "Not worse" tolerance 0.03 OVRL per window, and a light route for clean inputs.** The
  same window of an already clean laptop-mic recording (input OVRL 3.36) came out at 3.33-3.38
  depending only on chunk length and context; that spread is the resolution of the comparison. When
  the input already meets the podcast-mic target on a window that got "worse", the fix loop tries the
  mix-only route and keeps whichever round passes more gates.
- **D15b. SRMR is not part of the podcast-mic target.** Started as SRMR ≥ 9 from practice. On 96
  clean references SRMR spans 4-17 (EN p10 6.0 / median 11.3, RU p10 5.4 / median 7.3); a LibriTTS-R
  studio speaker reads 4.2 clean, and our restoration of that voice from a far, noisy, quiet room
  reached 4.65, drier than its studio original, while the old target called it a failure. Room echo
  even *raised* that voice's SRMR (clean 4.2, echoey 5.9; restored 5.1 with OVRL 3.11 → 3.49). So
  in "not worse" a lower SRMR counts only when OVRL did not rise by at least 0.10 on the same window
  (an added echo lowers both and still fails; tested).
- **D16. Sync gate is "not verified" against a very echoey input.** Envelope offsets against an
  echoey input scatter by several ms (see D7), so a 1 ms verdict would be noise. The gate passes at
  ≤1 ms when the points agree (IQR ≤ 1.5 ms, correlation ≥ 0.65, input SRMR ≥ 7) and fails beyond
  1 ms only then; with an unreliable measurement it says "not verified" whatever the number, and the
  fix loop never shifts the output by such a number (on a far, large-room input it once "corrected"
  a correctly timed track by a reverb-biased offset). The length must always be sample-exact.
- **D17. Loudness gate reads one channel of the dual-mono file.** ffmpeg's `-ac 1` downmix of a
  dual-mono file is 3 dB hotter than a channel (0.707·(L+R)); measuring the downmix as "stereo"
  double-counted 3 dB. A unit test pins both facts.
- **D18. Artifacts gate looks at seams, not at clicks in general.** Restored speech is full of sharp
  transients: an HF-spike click detector found ~60 "new clicks" per minute in Sidon output that
  the input did not have, and also flagged fricative onsets. A click at a seam is now a
  sample-level discontinuity at the exact seam sample (second difference vs its RMS over ±10 ms,
  limit adapted to the file); a background jump is a change, across the seam, of how loud the
  output's pauses are relative to the input's (the same input pauses, frame by frame, over the
  neighbouring chunks, at most 30 s each side), beyond 10 dB and beyond the file's own spread
  (median + 5 MAD over all seams). Simpler versions failed on real recordings: comparing pause floors
  of 2-10 s stretches gave 10-30 dB "jumps" wherever there were only half-second pauses in
  continuous speech. Seams of a source that fall outside the session timeline are ignored. Lost
  speech: input speech that is silent in the output.
  output. The test glues a room tone 15 dB louder after a seam and must fail.

- **D18a. Sources are aligned by GCC-PHAT, not by the envelope.** Envelope correlation against an
  echoey reference is pulled late by the echo (+5…+15 ms against synthetic rooms with a direct-to-
  reverberant ratio of 0 to -10 dB, +26 ms against an extreme one); the phase-transform
  cross-correlation lands on the direct sound (0.00 ms on the same rooms, +0.15 ms against a real
  OpenSLR room). On the real lavalier/screen pair it gave strong peaks, -6.04 ppm and a 0.31 ms fit
  residual. The envelope stays for the coarse search, the same-session score and as a fallback.
  The drift is fitted only over ≥ 300 s of points: on a 70 s clip the old fit turned 3 ms of noise
  into 103 ppm. GCC-PHAT does not work between a restored output and its input (Sidon regenerates the
  phase), so the sync gate keeps the envelope and says "not verified" on echoey inputs.
- **D18b. No non-finite sample reaches the mix.** A silent filler stretch made the level-match gain
  infinite and the delivered file full-scale noise; the gain is now taken only from finite loudness
  and limited to ±20 dB, and `_finish` refuses to mix anything with NaN/inf in it.

## Benchmark and thresholds

- **D19. Clean Russian reference: Russian LibriSpeech (public domain), not FLEURS.** FLEURS ru_ru
  (CC BY 4.0) was fetched first, but it is crowd-read in echoey rooms (SRMR 4-6), unusable as a
  clean reference. EN: LibriTTS-R test-clean (CC BY 4.0), not in Sidon's training data (Sidon
  trained on the train splits).
- **D20. Words in the benchmark are judged against what the recogniser hears in the clean
  original**, for both languages (RuLS transcripts are not streamed early in the archive).
- **D21. Target and gate thresholds from the benchmark** (`evals/results/thresholds.json`): OVRL ≥ 3.3
  stays (clean references p25 3.27, median 3.38); BAK goes from 3.9 to 4.0 (clean p25 4.08; the
  delivered lavalier track 3.9-4.1); the speaker threshold goes from 0.55 to 0.65, because two
  different clean speakers reached 0.633 while the same person after restoration never fell below
  0.667; the word margin stays 5 points (excess loss on correct restorations: p95 3 points).
