# What the numbers mean

All quality numbers are computed on 16 kHz mono, peak-normalised to 0.5, on **speech windows**
chosen by VAD (up to three 60 s windows: start, middle, end of the file, each the most
speech-dense stretch of its region). Pauses are never scored on their own: an echo tail that
drops in a pause says nothing about the echo inside the words.

| Metric | Range | Reads as | Notes |
|---|---|---|---|
| **SRMR** (speech-to-reverberation modulation energy ratio) | ~2-20 | higher = drier, less room | Good at echo. On an already dry close mic it can *fall* after restoration while quality rises, so for inputs with SRMR ≥ 10 it is checked against an absolute floor, not against the input. |
| **DNSMOS P.835** SIG / BAK / OVRL | 1-5 | predicted listener rating of speech / background / overall | Microsoft's no-reference model. Scored on 9.01 s segments with 1 s hop, averaged. Numbers are identical to torchmetrics' implementation. |
| **Confident-word loss** | 0-100 % | share of input words that the recogniser was sure about (prob ≥ 0.8, ≥ 3 letters) and that are missing or changed in the output | The gate subtracts the recogniser's own noise: the same input decoded with beam 5 vs greedy. Words the recogniser could not make out in a bad input are not counted, so a restored output that is understood *better* does not count as "changed". Plain WER is reported too. |
| **Speaker similarity** | -1..1 | cosine of ECAPA-TDNN embeddings, input vs output | Same person after restoration: ~0.87-0.95 in practice; another person: well below the threshold. |
| **Sync** | ms | offset of the output against the input on a 1 ms log-energy-envelope grid, median over 9 points | Reliable on clean and laptop recordings (±0.5 ms). Against a very echoey input the envelope itself is smeared and single windows scatter by several ms, so the gate says "not verified" there; the engine delay is proven separately on synthetic echo (see the benchmark). |
| **Loudness** | LUFS, dBTP | EBU R128 integrated loudness and true peak of the final stereo file | Target -16 LUFS ±0.5, parts within ±1 LU, true peak ≤ -1 dBTP. A dual-mono file reads 3 LU louder than its mono channel. |
| **Timbre** | dB per third octave | long-term average spectrum of speech frames, normalised at 400-2500 Hz | Gated only against a reference track (±1 dB per band, 60 Hz-16 kHz). |
| **Seams** | ratio | second difference at the exact seam sample vs its RMS over ±10 ms; background level step across the seam vs the same step in the input | The limits adapt to each file by sampling random places, because restored speech has transients of its own. |

Benchmark-only metrics (against a clean reference): **PESQ** (wide band) and **ESTOI**. SI-SDR is not
used for judging: Sidon regenerates the waveform, so sample-level error says little.
