# Changelog

## 0.3.0 — 2026-11-15

- **`--route natural`**: WPE dereverberation and a linear-phase EQ toward the timbre Sidon gives
  three short stretches of the same recording, then the usual mix. The voice is not
  re-synthesised; use it when a restored track sounds robotic. Opt-in, proven on one real laptop
  recording (accepted by ear), not for far camera mics (DECISIONS D22).
- A route forced with `--route` is kept by the probe and the fix loop; failed gates are reported
  (`--route master` used to deliver a Sidon track).
- The report says under every Sidon route that no gate detects a robotic sound and names the
  natural route; SKILL.md tells the agent to ask, not to blend tracks and not to answer
  "muffled" with a denoiser.
- WPE runs in double precision with a divergence guard; the engine revision is part of its cache key
  (a single-precision draft broke one 375 Hz bin on a real recording, pitfall 22).
- New pitfalls 17-22; tests in `tests/test_natural.py`, each checked to fail on a deliberately
  broken version (no subtraction, no chunk overlap, short last chunk merged, single precision, guard
  off, cache key without the engine revision, wrong probe offset, EQ not applied, EQ step missing,
  forced route swapped).

## 0.2.0 — 2026-11-02

- Renamed from `voice-rescue` to **Make Normal Voice** (`make-normal-voice`): the repository, plugin,
  skill, command, Python package, data folder and `MAKE_NORMAL_VOICE_HOME`. GitHub redirects the old
  repository address; an old install keeps working, a new one downloads the models into the new
  data folder (or `make-normal-voice install --from <old data folder>/models` reuses them).
- README: author links.

## 0.1.0 — 2026-11-02

First public release.

- `doctor`, `install` (pinned, checksummed, resumable model downloads), `analyze`, `plan`,
  `process`, `verify`, `align`, `match`, `ab`, `report`.
- Sidon restoration with sample-exact chunking: fixes the 5.6 ppm time squeeze of the reference
  recipe and removes the decoder's 7.375 ms delay.
- Compressor-free mix to −16 LUFS / −1 dBTP, 48 kHz 24-bit dual-mono.
- Multi-source sessions: envelope session matching, GCC-PHAT alignment to the direct sound, clock-drift removal, best source, gap
  filling, timbre matching, junction probe.
- Gates: podcast-mic target, not worse than input, confident words unchanged, same voice, sync and
  exact length, loudness and true peak, timbre vs reference, seams and lost speech. Each gate is
  tested against a deliberately broken output.
- Benchmark of eight engines and two hypothesis branches over twelve degradation classes in EN and RU.
- Memory bounded to about 5 GB (30 s chunks, 3 s decoder windows, own band-by-band SRMR), with a free-memory check before and during restoration (inside a Linux container, against the
  container's own limit).
- Checked by a clean install from GitHub on macOS and in an Ubuntu 24.04 container.
