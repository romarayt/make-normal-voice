# Troubleshooting

**`uv` not found.** Install it: `curl -LsSf https://astral.sh/uv/install.sh | sh` (or `brew install uv`).
The launcher uses a uv-managed Python 3.12, so a broken system Python does not matter.

**First run is slow.** The launcher creates the environment (about 1 GB of packages: torch,
transformers, onnxruntime, faster-whisper) in the data folder:
`~/Library/Application Support/voice-rescue/env` on macOS, `~/.local/share/voice-rescue/env` on
Linux (`VOICE_RESCUE_HOME` overrides). Later runs start in seconds.

**Models.** `voice-rescue install` downloads about 1.5 GB (Sidon 1 GB, Whisper small 0.5 GB,
DNSMOS and the speaker model 26 MB) and asks once. Downloads resume after an interruption; just
run the command again. Each file is checked against its sha256. A file that fails the check is
deleted, never used.

**Offline.** Everything that is already installed works without network. `voice-rescue doctor`
lists what is missing; `process` refuses to start without the restoration model and says which
command fetches it. `install --from <folder>` copies model files you already have (from another
machine or a Hugging Face cache) after checking their sha256.

**ffmpeg missing or too old.** `voice-rescue doctor` lists missing filters. macOS: `brew install
ffmpeg`. Debian/Ubuntu: `sudo apt-get install ffmpeg`. Windows: `winget install Gyan.FFmpeg`.

**Disk space.** One hour of audio needs about 4 GB of working space during processing; the
cached restoration output (≈0.7 GB per hour) is kept in the data folder's `cache/` so that a
re-run takes minutes. Delete `cache/` to free it.

**Memory.** Restoration peaks at about 4.5 GB (30 s chunks), the rest of a one-hour run at about
3-4 GB more. voice-rescue checks free memory before it starts and before every chunk: with too
little it shortens the chunks to 20 s (about 3.5 GB) or stops with a message instead of letting the
computer swap. `--chunk-seconds 20` forces the smaller setting. Do not run two long jobs at once.
In a Linux container the container's own memory limit (cgroup v1 or v2) counts, not the host's.

**GPU.** On a machine with an NVIDIA GPU and CUDA PyTorch, `voice-rescue install
sidon-feature-extractor-cuda sidon-decoder-cuda` and `--device cuda`. Untested in v0.1.0.

**Windows.** A PowerShell launcher is included (`scripts/voice-rescue.ps1`) but untested in v0.1.0.

**The result sounds wrong although the gates passed.** Trust your ear. Common reasons: the voice
timbre moved (restoration rebuilds the voice; try `--reference` with a track you liked), music or
another sound under the voice was removed or garbled (not supported in v0.1.0), two people talk at
once (not supported). Please open an issue with the numbers from `metrics.json` (not the audio).
