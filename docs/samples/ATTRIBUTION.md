# README example

- Speech: `tests/fixtures/speaker_b.flac`, cut from **LibriTTS-R** test-clean, speaker 1188
  (Koizumi et al., 2023), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
  <https://www.openslr.org/141/>.
- Room: real room impulse response `RVB2014_type1_rir_smallroom2_far_anglb` and room noise
  `RVB2014_type1_noise_mediumroom1_10` (REVERB challenge data in **OpenSLR 28 RIRS_NOISES**,
  Apache-2.0, <https://www.openslr.org/28/>), noise at 20 dB SNR, level set to -38 LUFS like a
  far camera mic.
- `before.mp3` is that degraded input, `after.mp3` the output of `make-normal-voice process`; the same
  30 s at equal loudness (MP3 96 kbps for size; the real output is 48 kHz / 24-bit WAV).
