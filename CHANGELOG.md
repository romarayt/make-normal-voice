# Changelog

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
