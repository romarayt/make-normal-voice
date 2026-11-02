"""voice-rescue command line: doctor, install, analyze, plan, process, verify, align, match, ab, report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def _print_json(d):
    print(json.dumps(d, indent=2, ensure_ascii=False, default=str))


def cmd_doctor(a):
    from . import install

    rep = install.doctor(verify_hashes=a.verify_hashes)
    if a.json:
        _print_json(rep)
    else:
        print(f"voice-rescue {__version__}; data folder: {rep['data_dir']}")
        for it in rep["items"]:
            mark = "ok " if it["ok"] else ("-- " if it.get("optional") else "NO ")
            fix = f"   → {it['fix']}" if not it["ok"] and it.get("fix") else ""
            print(f"  [{mark}] {it['name']}: {it['detail']}{fix}")
        print("ready" if rep["ok"] else "not ready: run `voice-rescue install` (and follow the hints above)")
    return 0 if rep["ok"] else 1


def cmd_install(a):
    from . import install

    ok = True
    if not a.skip_ffmpeg:
        ok = install.try_install_ffmpeg(a.yes) and ok
    ids = a.components or None
    if a.all:
        ids = list(install.COMPONENTS)
    ok = install.install(ids, yes=a.yes, from_dir=Path(a.from_dir).expanduser() if a.from_dir else None) and ok
    return 0 if ok else 1


def _need_models(yes: bool, ids=None) -> bool:
    from . import install

    miss = install.missing(ids or install.CORE)
    if not miss:
        return True
    print("Some models are missing: " + ", ".join(c.id for c in miss))
    return install.install([c.id for c in miss], yes=yes)


def cmd_analyze(a):
    from . import analyze

    if not _need_models(a.yes, ["dnsmos"]):
        return 1
    res = [analyze.analyze(Path(f), quick=a.quick) for f in a.files]
    if a.out:
        Path(a.out).write_text(json.dumps(res if len(res) > 1 else res[0], indent=2, ensure_ascii=False, default=str))
    if a.json or not a.out:
        _print_json(res if len(res) > 1 else res[0])
    return 0


def cmd_plan(a):
    from . import analyze, route

    if not _need_models(a.yes, ["dnsmos"]):
        return 1
    for f in a.files:
        p = analyze.analyze(Path(f))
        pl = route.plan(p, name=Path(f).name)
        print(pl.line(a.lang))
        for w in pl.warnings:
            print("  ! " + w)
        if a.json:
            _print_json({"plan": pl.to_dict(), "summary": analyze.summary(p)})
    return 0


def cmd_process(a):
    from . import pipeline

    if not _need_models(a.yes):
        return 1
    exp = {k.strip(): True for k in (a.experimental or "").split(",") if k.strip()}
    opts = pipeline.Options(
        out_dir=Path(a.out), reference=Path(a.reference) if a.reference else None,
        sync_to=Path(a.sync_to) if a.sync_to else None, trim_start=a.trim_start, trim_end=a.trim_end,
        trim_tail=a.trim_tail, target=a.target, device=a.device, asr_model=a.asr_model, language=a.language,
        probe=not a.no_probe, lang=a.lang, keep_work=a.keep_work, experimental=exp, force_route=a.route,
        name=a.name, chunk_s=a.chunk_seconds)
    doc = pipeline.process([Path(f) for f in a.files], opts)
    if doc.get("refused"):
        print(doc["reason"])
        return 2
    bad = [g for g in doc["verify"]["gates"] if g["status"] == "fail"]
    print(f"Output: {doc['output']}")
    print(f"Report: {Path(a.out) / 'report.md'}")
    for g in doc["verify"]["gates"]:
        print(f"  {g['status']:>10}  {g['name']}: {g['summary']}")
    return 0 if not bad else 3


def cmd_verify(a):
    from . import verify

    if not _need_models(a.yes):
        return 1
    res = verify.verify(Path(a.input), Path(a.output), reference=a.reference,
                        mapping=verify.Mapping(offset=a.offset, ppm=a.ppm), target=a.target,
                        asr_model=a.asr_model, language=a.language,
                        expected_len=a.expected_len)
    if a.json:
        _print_json(res)
    else:
        for g in res["gates"]:
            print(f"  {g['status']:>10}  {g['name']}: {g['summary']}")
    return 0 if res["ok"] else 3


def cmd_align(a):
    from . import align
    from .audio import load_mono

    x, y = load_mono(Path(a.a)), load_mono(Path(a.b))
    same, off = align.same_session(x, y)
    out = {"same_session": same, "envelope_match": round(off.score, 3), "coarse_offset_s": round(off.seconds, 3)}
    if a.drift:
        d = align.measure_drift(x, y, guess=off.seconds)
        out.update({"offset_s": round(d.offset, 5), "ppm": round(d.ppm, 3), "fit_residual_ms": round(d.residual_ms, 3)})
    _print_json(out)
    return 0


def cmd_match(a):
    from . import match
    from .audio import load_mono, write_wav

    ref, x = load_mono(Path(a.reference), 48000), load_mono(Path(a.file), 48000)
    y, log = match.match(ref, x, 48000, passes=a.passes)
    write_wav(Path(a.out), y, 48000, subtype="FLOAT")
    cs, d = match.residual(ref, y, 48000)
    _print_json({"passes": log, "residual_db": {str(int(c)): round(float(v), 2) for c, v in zip(cs, d)}})
    return 0


def cmd_ab(a):
    from . import report
    from .audio import dual_mono, load_channel, write_wav

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    after = out / "ab_after.wav"
    y = load_channel(Path(a.output), 48000)[int((a.start + a.offset) * 48000): int((a.start + a.offset + a.length) * 48000)]
    write_wav(after, dual_mono(y), 48000)
    report.loudness_matched_copy(Path(a.input), out / "ab_before.wav", start=a.start, duration=a.length, like=after)
    print(out / "ab_before.wav")
    print(after)
    return 0


def cmd_report(a):
    from . import report

    doc = json.loads(Path(a.metrics).read_text())
    md = report.render(doc, lang=a.lang)
    if a.out:
        Path(a.out).write_text(md)
    else:
        print(md)
    return 0


def build() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="voice-rescue", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("doctor", help="what is installed and what is missing")
    s.add_argument("--json", action="store_true")
    s.add_argument("--verify-hashes", action="store_true", help="re-check sha256 of every model file")
    s.set_defaults(fn=cmd_doctor)

    s = sub.add_parser("install", help="download missing models (asks once; --yes for unattended runs)")
    s.add_argument("components", nargs="*", help="component ids (default: the core set)")
    s.add_argument("--all", action="store_true", help="also optional components (CUDA weights, large ASR)")
    s.add_argument("--yes", "-y", action="store_true")
    s.add_argument("--from", dest="from_dir", help="take files from a local folder when their sha256 matches")
    s.add_argument("--skip-ffmpeg", action="store_true")
    s.set_defaults(fn=cmd_install)

    for name, fn, hlp in (("analyze", cmd_analyze, "passport of each recording"),
                          ("plan", cmd_plan, "passport + route, one line per track")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("files", nargs="+")
        s.add_argument("--json", action="store_true")
        s.add_argument("--yes", "-y", action="store_true")
        if name == "analyze":
            s.add_argument("--out")
            s.add_argument("--quick", action="store_true", help="skip SRMR")
        else:
            s.add_argument("--lang", default="en", choices=["en", "ru"])
        s.set_defaults(fn=fn)

    s = sub.add_parser("process", help="clean one track, or several sources of one session")
    s.add_argument("files", nargs="+")
    s.add_argument("--out", required=True)
    s.add_argument("--reference", help="delivered track whose timbre and loudness to match")
    s.add_argument("--sync-to", help="put the result on this file's timeline (e.g. the video)")
    s.add_argument("--trim-start", type=float)
    s.add_argument("--trim-end", type=float)
    s.add_argument("--trim-tail", action="store_true", help="cut after the last speech + 2 s (only on request)")
    s.add_argument("--target", type=float, default=-16.0, help="loudness target, LUFS")
    s.add_argument("--device", default="auto", help="auto | cpu | cuda")
    s.add_argument("--asr-model", default="small", choices=["small", "turbo"])
    s.add_argument("--language", help="speech language code for the word check, e.g. en, ru")
    s.add_argument("--no-probe", action="store_true")
    s.add_argument("--lang", default="en", choices=["en", "ru"], help="language of report.md")
    s.add_argument("--keep-work", action="store_true")
    s.add_argument("--route", choices=["sidon", "master"], help="force a route instead of the automatic one")
    s.add_argument("--experimental", help="comma list: declip,hum_notch")
    s.add_argument("--name", help="output file stem")
    s.add_argument("--chunk-seconds", type=float, default=30.0,
                   help="Sidon chunk length (default 30, ~5 GB peak); shortened automatically if memory is short")
    s.add_argument("--yes", "-y", action="store_true")
    s.set_defaults(fn=cmd_process)

    s = sub.add_parser("verify", help="gates: output vs input and vs the podcast-mic target")
    s.add_argument("input")
    s.add_argument("output")
    s.add_argument("--reference")
    s.add_argument("--offset", type=float, default=0.0, help="t_out = t_in + offset (after a trim, -trim_start)")
    s.add_argument("--ppm", type=float, default=0.0)
    s.add_argument("--expected-len", type=int, help="expected output length in 48 kHz samples")
    s.add_argument("--target", type=float, default=-16.0)
    s.add_argument("--asr-model", default="small", choices=["small", "turbo"])
    s.add_argument("--language")
    s.add_argument("--json", action="store_true")
    s.add_argument("--yes", "-y", action="store_true")
    s.set_defaults(fn=cmd_verify)

    s = sub.add_parser("align", help="are two recordings the same session; offset and clock drift")
    s.add_argument("a")
    s.add_argument("b")
    s.add_argument("--drift", action="store_true")
    s.set_defaults(fn=cmd_align)

    s = sub.add_parser("match", help="match timbre of FILE to REFERENCE (linear-phase EQ, no delay)")
    s.add_argument("reference")
    s.add_argument("file")
    s.add_argument("--out", required=True)
    s.add_argument("--passes", type=int, default=2)
    s.set_defaults(fn=cmd_match)

    s = sub.add_parser("ab", help="before/after listening probes at equal loudness")
    s.add_argument("input")
    s.add_argument("output")
    s.add_argument("--out", required=True)
    s.add_argument("--start", type=float, default=0.0)
    s.add_argument("--offset", type=float, default=0.0)
    s.add_argument("--length", type=float, default=30.0)
    s.set_defaults(fn=cmd_ab)

    s = sub.add_parser("report", help="re-render report.md from metrics.json")
    s.add_argument("metrics")
    s.add_argument("--lang", default="en", choices=["en", "ru"])
    s.add_argument("--out")
    s.set_defaults(fn=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build().parse_args(argv)
    try:
        return a.fn(a) or 0
    except KeyboardInterrupt:
        return 130
    except Exception as e:  # noqa: BLE001 - one readable line for the agent, traceback with VOICE_RESCUE_DEBUG
        import os

        if os.environ.get("VOICE_RESCUE_DEBUG"):
            raise
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
