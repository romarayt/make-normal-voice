"""Fetch the open evaluation material, partially and resumably, into $VR_EVAL_DATA.

* English clean speech: LibriTTS-R test-clean (CC BY 4.0), streamed from OpenSLR 141; stops after
  enough utterances from enough speakers (no need to download the whole 1.3 GB archive).
* Russian clean speech: LibriVox readings (public domain), one book per reader, partial MP3 downloads
  from archive.org. (Russian LibriSpeech, OpenSLR 96, is the same material but its 9 GB archive
  streamed too slowly to reach more than one reader.)
  (FLEURS ru_ru, CC BY 4.0, was tried first: crowd-read in echoey rooms, SRMR 4-6, not a clean reference.)
* Real room impulse responses and isotropic noises: OpenSLR 28 RIRS_NOISES (Apache-2.0), read
  member-by-member from the remote zip with HTTP Range requests.

Nothing from here goes into the repository: it is downloaded on demand, for local evaluation.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import tarfile
import urllib.request
import zipfile
from pathlib import Path

LIBRITTS_R = "https://www.openslr.org/resources/141/test_clean.tar.gz"
FLEURS_TAR = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/ru_ru/audio/test.tar.gz"
FLEURS_TSV = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/ru_ru/test.tsv"
RIRS_ZIP = "https://www.openslr.org/resources/28/rirs_noises.zip"
RULS_TAR = "https://openslr.elda.org/resources/96/ruls_data.tar.gz"


def data_dir() -> Path:
    d = Path(os.environ.get("VR_EVAL_DATA", Path.home() / ".cache" / "voice-rescue-evals")).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _open(url: str, timeout: float = 60.0):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "voice-rescue-evals"}), timeout=timeout)


def fetch_libritts(n_speakers: int = 10, per_speaker: int = 3, min_s: float = 7.0, max_s: float = 20.0) -> Path:
    out = data_dir() / "clean_en"
    out.mkdir(exist_ok=True)
    meta_p = out / "meta.json"
    if meta_p.exists() and len(json.loads(meta_p.read_text())) >= n_speakers * per_speaker:
        return out
    import soundfile as sf

    got: dict[str, list[dict]] = {}
    texts: dict[str, str] = {}
    pending: dict[str, bytes] = {}
    with _open(LIBRITTS_R) as r, tarfile.open(fileobj=r, mode="r|gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            name = m.name.rsplit("/", 1)[-1]
            if name.endswith(".normalized.txt"):
                uid = name[: -len(".normalized.txt")]
                texts[uid] = tf.extractfile(m).read().decode("utf-8").strip()
            elif name.endswith(".wav"):
                uid = name[:-4]
                spk = uid.split("_")[0]
                if len(got) >= n_speakers and spk not in got:
                    if all(len(v) >= per_speaker for v in got.values()):
                        break
                    continue
                if len(got.get(spk, [])) >= per_speaker:
                    continue
                pending[uid] = tf.extractfile(m).read()
            for uid in [u for u in pending if u in texts]:
                data = pending.pop(uid)
                info = sf.info(io.BytesIO(data))
                if not (min_s <= info.duration <= max_s):
                    continue
                spk = uid.split("_")[0]
                (out / f"{uid}.wav").write_bytes(data)
                got.setdefault(spk, []).append({"id": uid, "speaker": spk, "text": texts[uid],
                                                "duration": info.duration, "lang": "en"})
                print(f"  en {uid} {info.duration:.1f}s ({len(got)} speakers)", flush=True)
    meta = [u for v in got.values() for u in v]
    meta_p.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    return out


def fetch_fleurs(n: int = 60, min_s: float = 7.0, max_s: float = 20.0) -> Path:
    out = data_dir() / "clean_ru_pool"
    out.mkdir(exist_ok=True)
    meta_p = out / "meta.json"
    if meta_p.exists() and len(json.loads(meta_p.read_text())) >= n:
        return out
    import soundfile as sf

    with _open(FLEURS_TSV) as r:
        rows = list(csv.reader(io.TextIOWrapper(r, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE))
    by_file = {row[1]: row for row in rows if len(row) > 6}
    meta = []
    with _open(FLEURS_TAR) as r, tarfile.open(fileobj=r, mode="r|gz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".wav"):
                continue
            fname = m.name.rsplit("/", 1)[-1]
            row = by_file.get(fname)
            if row is None:
                continue
            data = tf.extractfile(m).read()
            info = sf.info(io.BytesIO(data))
            if not (min_s <= info.duration <= max_s):
                continue
            uid = "ru_" + fname[:-4]
            (out / f"{uid}.wav").write_bytes(data)
            # FLEURS has no speaker ids; gender + sentence id is the best grouping available
            meta.append({"id": uid, "speaker": f"fleurs-{row[0]}-{row[6]}", "text": row[2], "duration": info.duration,
                         "lang": "ru", "gender": row[6]})
            print(f"  ru {uid} {info.duration:.1f}s ({len(meta)})", flush=True)
            if len(meta) >= n:
                break
    meta_p.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    return out


def fetch_ruls(n: int = 30, per_book: int = 2, per_reader: int = 6, min_s: float = 7.0, max_s: float = 20.0,
               max_bytes: float = 3e9) -> Path:
    """Russian LibriSpeech (public domain LibriVox readings): close-mic home recordings, cleaner than
    FLEURS. Streams the archive and stops early; transcripts are not needed (the benchmark compares
    recognised words of the output with those of the clean original). Re-runs keep what was saved."""
    out = data_dir() / "clean_ru"
    out.mkdir(exist_ok=True)
    meta_p = out / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else []
    if len(meta) >= n:
        return out
    import soundfile as sf

    have = {m["id"] for m in meta}
    per_b, per_r = {}, {}
    for m in meta:
        per_b[m["book"]] = per_b.get(m["book"], 0) + 1
        per_r[m["speaker"]] = per_r.get(m["speaker"], 0) + 1
    with _open(RULS_TAR, timeout=180) as r:
        counted = _Counting(r)
        with tarfile.open(fileobj=counted, mode="r|gz") as tf:
            for m in tf:
                if counted.n > max_bytes or len(meta) >= n:
                    break
                if not m.isfile() or not m.name.endswith((".wav", ".flac")):
                    continue
                parts = m.name.split("/")
                book = "/".join(parts[-3:-1]) if len(parts) >= 3 else parts[0]  # <split>/audio/<reader>/<book>/<file>
                reader = "ruls-" + (parts[-3] if len(parts) >= 3 else parts[0])
                uid = "ru_" + parts[-1].rsplit(".", 1)[0]
                if uid in have or per_b.get(book, 0) >= per_book or per_r.get(reader, 0) >= per_reader:
                    continue
                data = tf.extractfile(m).read()
                info = sf.info(io.BytesIO(data))
                if not (min_s <= info.duration <= max_s):
                    continue
                x, sr = sf.read(io.BytesIO(data), dtype="float32")
                sf.write(out / f"{uid}.wav", x, sr, subtype="PCM_16")
                per_b[book] = per_b.get(book, 0) + 1
                per_r[reader] = per_r.get(reader, 0) + 1
                meta.append({"id": uid, "speaker": reader, "book": book, "text": "", "duration": info.duration,
                             "lang": "ru", "path_in_archive": m.name})
                meta_p.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
                print(f"  ru {uid} {reader} {info.duration:.1f}s ({len(meta)}, {counted.n / 1e6:.0f} MB read)", flush=True)
    return out


LIBRIVOX_RU = ["chortova_kukla_2211_librivox", "belkintales_1406_librivox", "childhood_2312_librivox",
               "obryv_1906_librivox", "on_the_edge_of_the_world_1902_librivox", "lampada_1810_librivox",
               "stepnyyeskazki_2204_librivox", "vekhi_2011_librivox"]


def fetch_librivox_ru(per_item: int = 2, head_bytes: int = 4_000_000, skip_s: float = 40.0) -> Path:
    """Russian LibriVox readings (public domain), one book per reader, from archive.org.

    Only the first ~4 MB of the first 64 kbps chapter MP3 is fetched (HTTP Range); the LibriVox intro
    is skipped and speech spans of 8-15 s are cut at pauses found by VAD.
    """
    import soundfile as sf

    sys_path = Path(__file__).resolve().parents[1] / "skills" / "voice-rescue" / "src"
    import sys as _sys
    _sys.path.insert(0, str(sys_path))
    from voice_rescue import vad
    from voice_rescue.audio import load_mono

    out = data_dir() / "clean_ru"
    out.mkdir(exist_ok=True)
    meta_p = out / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else []
    done = {m["speaker"] for m in meta}
    for ident in LIBRIVOX_RU:
        if f"librivox-{ident}" in done:
            continue
        with _open(f"https://archive.org/metadata/{ident}") as r:
            md = json.loads(r.read())
        mp3s = sorted(f["name"] for f in md.get("files", []) if f["name"].endswith("_64kb.mp3"))
        if not mp3s:
            continue
        url = f"https://archive.org/download/{ident}/{urllib.request.quote(mp3s[0])}"
        req = urllib.request.Request(url, headers={"Range": f"bytes=0-{head_bytes - 1}", "User-Agent": "voice-rescue-evals"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        tmp = out / f"_{ident}.mp3"
        tmp.write_bytes(data)
        x = load_mono(tmp, 16000)
        x48 = load_mono(tmp, 48000)
        tmp.unlink()
        segs = [(a, b) for a, b in vad.speech_segments(x) if a >= skip_s]
        spans, cur = [], None
        for a, b in segs:
            if cur and a - cur[1] < 0.6 and b - cur[0] <= 15.0:
                cur = (cur[0], b)
            else:
                if cur and cur[1] - cur[0] >= 8.0:
                    spans.append(cur)
                cur = (a, b)
        picked = spans[: per_item]
        for k, (a, b) in enumerate(picked):
            uid = f"ru_lv_{ident.split('_librivox')[0]}_{k}"
            seg = x48[int((a - 0.15) * 48000): int((b + 0.15) * 48000)]
            sf.write(out / f"{uid}.wav", seg, 48000, subtype="PCM_16")
            meta.append({"id": uid, "speaker": f"librivox-{ident}", "text": "", "duration": len(seg) / 48000,
                         "lang": "ru", "source": url, "start_s": round(a, 2)})
            print(f"  ru {uid} {len(seg) / 48000:.1f}s", flush=True)
        meta_p.write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    return out


class _Counting(io.RawIOBase):
    def __init__(self, f):
        self.f, self.n = f, 0

    def readable(self):
        return True

    def readinto(self, b):
        data = self.f.read(len(b))
        b[: len(data)] = data
        self.n += len(data)
        return len(data)


class HttpFile(io.RawIOBase):
    """Seekable read-only view of a remote file through HTTP Range requests (for zipfile)."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        with _open(url) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}", "User-Agent": "voice-rescue-evals"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def fetch_rirs(max_rirs: int = 40, max_noises: int = 12) -> Path:
    out = data_dir() / "rirs"
    out.mkdir(exist_ok=True)
    meta_p = out / "meta.json"
    if meta_p.exists():
        return out
    zf = zipfile.ZipFile(io.BufferedReader(HttpFile(RIRS_ZIP), buffer_size=1 << 20))
    names = zf.namelist()
    real = [n for n in names if "/real_rirs_isotropic_noises/" in n and n.endswith(".wav")]
    base = {n: n.rsplit("/", 1)[-1].lower() for n in real}
    noises = [n for n in real if "_noise_" in base[n]]
    # reverberant room responses only (the RWCP "ane" set is anechoic)
    rirs = [n for n in real if "_noise_" not in base[n] and "_ane_" not in base[n]]
    # spread the picks over the different collections (RWCP, REVERB, AIR ...)
    def spread(lst, k):
        lst = sorted(lst)
        step = max(1, len(lst) // k)
        return lst[::step][:k]
    meta = {"rirs": [], "noises": []}
    for kind, lst, k in (("rirs", rirs, max_rirs), ("noises", noises, max_noises)):
        for n in spread(lst, k):
            dst = out / kind / n.rsplit("/", 1)[-1]
            dst.parent.mkdir(exist_ok=True)
            if not dst.exists():
                dst.write_bytes(zf.read(n))
            meta[kind].append(dst.name)
            print(f"  {kind} {dst.name}", flush=True)
    meta_p.write_text(json.dumps(meta, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="*", default=["en", "ru", "rirs"])  # also: fleurs
    a = ap.parse_args()
    if "en" in a.what:
        print("LibriTTS-R test-clean ->", fetch_libritts())
    if "ru" in a.what:
        print("Russian LibriVox ->", fetch_librivox_ru())
    if "ruls" in a.what:
        print("Russian LibriSpeech ->", fetch_ruls())
    if "fleurs" in a.what:
        print("FLEURS ru test ->", fetch_fleurs())
    if "rirs" in a.what:
        print("OpenSLR 28 ->", fetch_rirs())


if __name__ == "__main__":
    main()
