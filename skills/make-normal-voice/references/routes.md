# Track classes and routes

`make-normal-voice plan` prints the class and the route of every track in one line. Classes are decided
on the medians of the analysis windows (see `thresholds.md` for every number).

| Class | Detected by | Route | Status |
|---|---|---|---|
| Far mic, room echo (camera) | SRMR < 8 or OVRL < 2.8 | Sidon → mix | proven: real camera recordings; benchmark OVRL 1.84 → 3.46 (echo), 1.62 → 3.50 (echo + noise) |
| Built-in laptop mic | SRMR < 13 and OVRL < 3.1 | Sidon → mix | proven: real screen recordings; benchmark 2.42 → 3.49 |
| Close mic, noisy after gain | BAK < 3.9 | Sidon → mix | proven: real lavalier; benchmark (5 dB SNR) 2.37 → 3.51; a classic denoiser keeps the waveform closer (PESQ 2.6 vs 2.1) but is less clean (3.34) |
| Clean close mic | OVRL ≥ 3.3, BAK ≥ 4.0, SRMR ≥ 10 | mix only; Sidon when it is glued to restored parts | proven: benchmark mix-only keeps PESQ 4.55 / ESTOI 1.00; Sidon would regenerate it (PESQ 2.70) |
| In between | anything else | Sidon → mix | partly proven |
| Parts of different loudness | speech loudness of ≥ 60 s stretches differs by > 1 LU | after Sidon, parts are re-measured and levelled, the switch sits in a pause | proven (Sidon already evens most of it out) |
| Several sources of one session | envelope match ≥ 0.6 against the timeline | best source by OVRL; others fill the gaps; aligned to 1 ms, drift removed, timbre matched, junction probe | proven |
| Long recordings (> 6 min) | duration | probe on the hardest 150 s first; a failed light route on the probe switches to Sidon | proven |
| Clipping | > 0.01 % samples at full scale or flat tops | Sidon as usual (benchmark 3.23 → 3.45, PESQ 2.37 → 2.62); `adeclip` first only with `--experimental declip` | Sidon alone: benchmark; the declip branch: see README |
| Mains hum 50/60 Hz | harmonic peaks > 10 dB in pauses | Sidon as usual (benchmark 3.28 → 3.48, BAK 4.21); notch first only with `--experimental hum_notch` | Sidon alone: benchmark; the notch branch: see README |
| Narrow band, phone, low-bitrate codec | spectrum ends below 8.6 kHz | Sidon (rebuilds 48 kHz): phone 3.32 → 3.46, MP3 24 kbps 3.37 → 3.44 | benchmark; inputs that already meet the target are only mixed |
| Music or other tonal sound under the voice | loud tonal pauses | warned; restoration removes the music (benchmark: DNSMOS then rates it clean speech, 2.71 → 3.49) | limitation |
| Two voices at once | not detected in v0.2.0 | — | limitation |
| Digital dropouts | runs of exact zeros > 10 ms | reported; patched only when another source of the session has the speech | partly |
| Unreadable file | no decodable audio stream | refused with ffmpeg's error | — |

The router never chooses a compressor, a stereo widener or a different engine on its own: those
were tried on real recordings and lost (see `pitfalls.md`).
