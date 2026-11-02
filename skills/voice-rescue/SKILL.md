---
name: voice-rescue
description: >-
  Turns any speech recording (camera audio with room echo, a laptop's built-in mic, a noisy or
  bad mic, a screen recording, several devices of one session) into an edit-ready voice track that
  sounds close-miked, like a podcast microphone, fully locally. Analyses each track, picks a route
  (Sidon speech restoration + a compressor-free mix), verifies the result with measurements (DNSMOS,
  SRMR, word check, same-voice check, 1 ms sync, loudness) and fixes what fails. Use when the user
  asks to clean up audio, remove echo or reverb, remove background noise from speech, fix a voice
  recording, make it sound like a podcast mic or a studio mic, align a lavalier with camera audio,
  or says "почисти звук", "убери эхо", "убери шум", "сделай звук как с микрофона", "звук как из
  туалета", "сделай голос как на подкасте", "выровняй звук с петлички под камеру".
license: MIT
compatibility: >-
  macOS or Linux with ffmpeg and uv, about 6 GB of free memory. Runs on CPU (about 5x faster than
  real time on Apple M4 Max); first install downloads about 1.5 GB of models and about 1 GB of Python
  packages. No cloud APIs.
metadata:
  version: "0.1.0"
  repository: "https://github.com/romarayt/voice-rescue"
---

# voice-rescue

Clean speech recordings into a close-mic, podcast-style voice track. Everything runs on the
user's machine; their audio is never uploaded anywhere.

`VR` below means the launcher in this skill folder: `${CLAUDE_SKILL_DIR}/scripts/voice-rescue`
(in agents without that variable, `scripts/voice-rescue` inside the folder that holds this
SKILL.md). It creates its own locked Python 3.12 environment on first run (uv), so do not use the
system Python or pip.

## Workflow

1. **Check the machine.** `VR doctor`. If models are missing, tell the user what will be
   downloaded and how big it is (doctor prints it), ask once, then `VR install` (add `--yes` only
   if the user already agreed or asked for an unattended run). Missing ffmpeg: `VR install`
   offers `brew`/`winget`; on Linux tell the user the `apt` command, never run sudo yourself.
2. **Explain the route before processing.** `VR plan <files>` prints one line per track, e.g.
   `camera.wav: echo (SRMR 5.9, OVRL 2.53), quiet (-40 LUFS) → Sidon → mix`. Relay it to the user
   in their language (`--lang ru` prints it in Russian) together with any warnings.
3. **Process.** `VR process <files> --out <folder> [--lang ru] [--language ru]`.
   - Long recordings: run it in the background and relay the progress lines. A 20-minute track
     takes about 5 minutes on a fast CPU; an hour about 15. Files longer than 6 minutes are
     probed first on their hardest 2.5 minutes (`probe/` folder), then processed in full.
   - One job at a time: a run needs about 5 GB of memory at its peak. Never start two `process`
     runs in parallel; voice-rescue checks free memory and stops cleanly instead of making the
     computer swap.
   - Several files of one session (camera + screen recording + lavalier): pass them all;
     the skill checks they belong together by the speech envelope, aligns them to 1 ms, removes
     clock drift, takes the best source, fills gaps from the others and writes a junction probe.
     `--sync-to video.mp4` puts the result on the video's timeline.
   - `--reference delivered.wav` matches timbre and loudness to a track the user already has.
   - Trim only when the user asks: `--trim-tail` (after the last phrase + 2 s), `--trim-start`,
     `--trim-end`. Otherwise the track keeps the source length and start, so it drops under the
     video without re-syncing.
4. **Report honestly.** Read `<out>/report.md` (and `metrics.json` if you need numbers) and tell
   the user: what the recording was, what was done, before → after numbers, every gate that
   failed or could not be verified, what was cut (if anything), and the paths of
   `ab_before_30s.wav` / `ab_after_30s.wav` (same 30 s at equal loudness) so they can listen.

## Rules

- **You cannot hear.** Never write "sounds good/clean". Write "by measurement" and hand over the
  before/after probes. The listener's ear outranks every number; if they say it sounds bad, it is bad.
- **Never upload the user's audio**, never send it to an API, never commit it to a repository.
- **Deliver the whole track.** Same length, same start, sample-exact, unless the user asked for a
  trim. Do not cut by an edit plan: people sync the track to the picture themselves.
- **A failed gate is reported, not hidden.** After three fix rounds the track is still delivered,
  and the report says which gate failed, where and by how much.
- **Restoration rebuilds the voice.** Timbre can shift slightly; say so if the user is picky
  about their voice, and offer `--reference` matching to an earlier delivered track.
- Do not add a compressor, a stereo widener or "loudness maximising": they bring the room back
  or smear the phase (see `references/pitfalls.md`).

## Commands

| Command | What it does |
|---|---|
| `VR doctor [--json]` | what is installed, what is missing, disk space |
| `VR install [ids] [--yes] [--all]` | download missing models (pinned revisions, sha256-verified, resumable) |
| `VR analyze <files> [--json]` | passport: format, level per minute, clipping, dropouts, clicks, noise, hum, band, stereo, SRMR/DNSMOS on speech windows |
| `VR plan <files> [--lang ru]` | passport + route, one line per track |
| `VR process <files> --out DIR` | full run with probe, verify and up to 3 fix rounds |
| `VR verify <input> <output>` | the gates on any pair of files |
| `VR align <a> <b> --drift` | same session? offset and clock drift (ppm) |
| `VR match <ref> <file> --out` | timbre match, linear phase, zero delay |
| `VR ab <input> <output> --out DIR` | before/after probes at equal loudness |

Exit codes: 0 all gates passed, 3 delivered with failed gates, 1 error, 2 file unreadable.

## Reading further (only when needed)

- `references/routes.md`: track classes, how each is detected, what route it gets, which are proven.
- `references/thresholds.md`: every threshold and where it comes from.
- `references/metrics.md`: what SRMR, DNSMOS, WER noise floor, speaker similarity and sync mean.
- `references/pitfalls.md`: mistakes already made on real recordings and how the code guards against them.
- `references/troubleshooting.md`: install problems, offline use, slow downloads, disk space.
