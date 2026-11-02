# Thresholds and where they come from

All values live in `src/voice_rescue/thresholds.py`. "Practice" means real recordings processed by
hand before this tool existed (a camera with room echo, a laptop mic during screen recording, a
lavalier); "benchmark" means `evals/` on open speech (`evals/results/`).

## Podcast-mic target (quality gate)

| Metric | Threshold | Source |
|---|---|---|
| DNSMOS OVRL (median of the speech windows) | ≥ 3.3 | benchmark: 96 clean references (80 EN LibriTTS-R, 16 RU LibriVox) have OVRL p25 3.27, median 3.38; practice: a raw close lavalier scored 3.08-3.29, the best restored results 3.29-3.49 |
| DNSMOS BAK | ≥ 4.0 | benchmark: clean references BAK p25 4.08, median 4.15; practice: the raw lavalier after gain 3.6-3.9 (audible room noise), restored 3.9-4.1 |
| SRMR | not a target | it depends on the voice and delivery as much as on the room: on 96 clean references SRMR spans 4-17 (EN p10 6.0, median 11.3; RU p10 5.4, median 7.3), and one LibriTTS-R studio speaker sits at 4.2. Echo is caught by "not worse" (SRMR must rise on wet inputs) and by DNSMOS |

## Not worse than the input

| Rule | Value | Source |
|---|---|---|
| OVRL may not drop on any window | tolerance 0.03 | the spread of one window between equivalent restorations (chunk lengths 20-96 s, context 3-6 s): ±0.03 |
| SRMR may not drop | for inputs with SRMR < 10, unless OVRL rose by ≥ 0.10 on that window | practice: on dry close mics SRMR fell after restoration (12.4 → 11.4, 10.8 → 9.3) while quality rose; benchmark: for one LibriTTS-R voice room echo *raised* SRMR (clean 4.2, echoey 5.9) and restoration brought it back to 5.1 while OVRL went 3.11 → 3.49 |
| SRMR floor for dry inputs | ≥ 9, with the same OVRL exception | same |

## Router classes

| Class | Rule | Source |
|---|---|---|
| far mic, echo | SRMR < 8 or OVRL < 2.8 | practice: camera 4.7-9.2 / 2.43-2.65 |
| laptop mic | SRMR < 13 and OVRL < 3.1 | practice: 10.0-13.2 / 2.49-3.13 |
| noisy close mic | BAK < 3.9 | practice |
| clean close mic | OVRL ≥ 3.3, BAK ≥ 4.0, SRMR ≥ 10 | practice + benchmark (same as the target) |
| parts of different loudness | > 1 LU between ≥ 60 s stretches | practice: a screen recording whose first 2 minutes had no external mic, 6 dB quieter |
| clipping | > 0.01 % samples at full scale, or > 200 flat-top samples | conservative; route stays the default unless the benchmark shows a win |
| hum | harmonic prominence > 10 dB in pauses | same |
| narrow band | spectrum ends below 8.6 kHz | phone 4 kHz, 8 kHz codecs |

## Deliverable

| Gate | Value | Source |
|---|---|---|
| loudness | −16 ± 0.5 LUFS integrated on the stereo file; stretches within ±1 LU; true peak ≤ −1 dBTP | spoken-word streaming practice; the limiter at 0.84 gives about −1.5 dBTP |
| sync | median ≤ 1 ms over 9 points; length exact to the sample | a video frame at 25 fps is 40 ms; 1 ms is far below anything audible as lip-sync error |
| sync reliability | median correlation ≥ 0.65, IQR ≤ 1.5 ms, input SRMR ≥ 7; otherwise "not verified" (fail beyond 5 ms) | practice: offsets against an echoey camera input scatter 1-8 ms per window |
| timbre vs reference | ±1 dB per third octave, 60 Hz-16 kHz | practice: two matching passes left ±0.4 dB |

## Words and voice

| Gate | Value | Source |
|---|---|---|
| same voice | ECAPA cosine ≥ 0.65 | benchmark: the most similar pair of *different* clean speakers 0.633 (p99 0.578); the same person after restoration, against the clean original, never below 0.667 (p1 0.677) |
| words | lost confident words ≤ recogniser noise + 5 points | benchmark: on 132 restored windows the excess loss had median 0, p90 2.4, p95 3.0 points (max 18 on one far, noisy English clip); one replaced word in a 5.6 s clip is 7-10 points |

## Long recordings

Probe 150 s around the window with the lowest OVRL when the file is longer than 6 minutes.
