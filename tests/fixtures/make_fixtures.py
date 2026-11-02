"""Rebuild the test fixtures from LibriTTS-R test-clean (CC BY 4.0), fetched by evals/fetch_data.py."""

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evals"))
from fetch_data import data_dir  # noqa: E402

HERE = Path(__file__).parent
SPEC = {
    "speaker_a.flac": ["4507_16021_000020_000000", "4507_16021_000023_000008", "4507_16021_000033_000000",
                       "4507_16021_000025_000001", "4507_16021_000016_000003"],
    "speaker_b.flac": ["1188_133604_000046_000003", "1188_133604_000038_000003", "1188_133604_000050_000003",
                       "1188_133604_000067_000002"],
    "speaker_a_short.flac": ["4507_16021_000040_000000"],
}


def main():
    src = data_dir() / "clean_en"
    meta = {m["id"]: m for m in json.loads((src / "meta.json").read_text())}
    lines = []
    for name, ids in SPEC.items():
        parts, sr = [], None
        for uid in ids:
            x, sr = sf.read(src / f"{uid}.wav", dtype="float32")
            parts += [x, np.zeros(int(0.4 * sr), np.float32)]
            lines.append(f"| `{name}` | LibriTTS-R test-clean `{uid}` | {meta[uid]['text']} |")
        y = np.concatenate(parts[:-1])
        if name == "speaker_a_short.flac":
            y = y[: int(5.6 * sr)]
        sf.write(HERE / name, y, sr, subtype="PCM_16")
        print(name, f"{len(y) / sr:.1f} s")
    (HERE / "ATTRIBUTION.md").write_text(
        "# Test fixtures\n\nBuilt by `make_fixtures.py` from **LibriTTS-R** (Koizumi et al., 2023), test-clean subset, "
        "licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), "
        "distributed at <https://www.openslr.org/141/>. LibriTTS-R is derived from LibriTTS / LibriSpeech "
        "(public-domain LibriVox recordings). Utterances were concatenated with 0.4 s of silence; "
        "`speaker_a_short.flac` is cut to 5.6 s.\n\n| File | Source utterance | Text |\n|---|---|---|\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
