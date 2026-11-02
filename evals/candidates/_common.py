"""Shared CLI for candidate runners: `python <runner>.py inputs.json out_dir` (one venv per candidate)."""

import json
import sys
import time
from pathlib import Path


def run(process_one, setup=None):
    inputs = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    state = setup() if setup else None
    t_setup = time.time() - t0
    timing = {}
    for p in inputs:
        dst = out / Path(p).name
        if dst.exists():
            continue
        t = time.time()
        try:
            process_one(state, p, str(dst))
            timing[Path(p).name] = time.time() - t
        except Exception as e:  # noqa: BLE001 - a failing candidate is a benchmark result, not a crash
            timing[Path(p).name] = None
            print(f"FAILED {p}: {type(e).__name__}: {e}", flush=True)
        print(f"{Path(p).name} {timing[Path(p).name]}", flush=True)
    prev = out / "timing.json"
    old = json.loads(prev.read_text()) if prev.exists() else {}
    old.update(timing)
    old["_setup_s"] = t_setup
    prev.write_text(json.dumps(old, indent=1))
