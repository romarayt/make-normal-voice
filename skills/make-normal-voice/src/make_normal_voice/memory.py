"""How much memory the machine can still give, so a long run never pushes it into swap.

Measured peaks on CPU (Sidon, 16 kHz in, 48 kHz out): the feature extractor grows with the square
of the chunk length (+0.5 GB at 20 s, +1.8 GB at 40 s, +4.1 GB at 60 s, ~+12 GB at the reference
demo's 96 s), the decoder with its window (about 0.2 GB per second, so it runs on 3 s windows).
With 30 s chunks a whole Sidon run peaks at about 4.5 GB.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess

GB = 2**30
# peak resident memory of a Sidon worker for a given chunk length (s), measured, with margin
SIDON_PEAK_GB = {20: 3.5, 30: 5.0, 48: 8.5}


def available_gb() -> float | None:
    """Memory that can be used without swapping, in GB; None if it cannot be determined."""
    try:
        if platform.system() == "Darwin":
            # the kernel's own estimate first (it counts reclaimable file cache), then raw page counts
            mp = subprocess.run(["/usr/bin/memory_pressure"], capture_output=True, text=True, timeout=10).stdout
            m = re.search(r"free percentage:\s*([\d.]+)%", mp)
            if m:
                total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout)
                return float(m.group(1)) / 100 * total / GB
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
            page = int(re.search(r"page size of (\d+)", out).group(1))
            vals = {k.strip(): int(v.strip().rstrip(".")) for k, v in
                    (ln.split(":", 1) for ln in out.splitlines()[1:] if ":" in ln) if v.strip().rstrip(".").isdigit()}
            pages = sum(vals.get(k, 0) for k in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"))
            return pages * page / GB
        if os.path.exists("/proc/meminfo"):
            host = None
            with open("/proc/meminfo") as fh:
                for ln in fh:
                    if ln.startswith("MemAvailable:"):
                        host = int(ln.split()[1]) * 1024 / GB
            # inside a container /proc/meminfo shows the whole host; the container's own limit is the real one
            cg = _cgroup_free_gb()
            if host is None or cg is None:
                return host if cg is None else cg
            return min(host, cg)
    except Exception:  # noqa: BLE001 - unknown is an answer too
        return None
    return None


def _read(path: str) -> str | None:
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def _cgroup_free_gb(root: str = "/sys/fs/cgroup") -> float | None:
    """Limit minus usage (without reclaimable file cache) of this process's memory cgroup, v2 or v1."""
    for limit_f, usage_f, stat_f, cache_key in (
            ("memory.max", "memory.current", "memory.stat", "inactive_file"),
            ("memory/memory.limit_in_bytes", "memory/memory.usage_in_bytes", "memory/memory.stat",
             "total_inactive_file")):
        limit, usage = _read(f"{root}/{limit_f}"), _read(f"{root}/{usage_f}")
        if limit is None or usage is None:
            continue
        if not limit.isdigit() or int(limit) >= 2**60:  # "max" or v1's "no limit" sentinel
            return None
        cache = 0
        for ln in (_read(f"{root}/{stat_f}") or "").splitlines():
            k, _, v = ln.partition(" ")
            if k == cache_key and v.isdigit():
                cache = int(v)
        return max(0, int(limit) - int(usage) + cache) / GB
    return None


class NotEnoughMemory(RuntimeError):
    pass


def pick_chunk_seconds(requested: float | None) -> float:
    """The requested Sidon chunk if it fits in memory, else a shorter one; refuse if nothing fits."""
    avail = available_gb()
    want = requested or 30.0
    if avail is None:
        return want
    for s in sorted({want, 30.0, 20.0}, reverse=True):
        if s > want:
            continue
        need = SIDON_PEAK_GB.get(int(s), 0.13 * s + 1.5)
        if need <= avail * 0.8:
            return s
    raise NotEnoughMemory(
        f"only {avail:.1f} GB of memory is free; restoration needs about {SIDON_PEAK_GB[20]:.1f} GB. "
        "Close other applications and run again.")


def check_during_run(min_gb: float = 1.5) -> None:
    """Abort cleanly instead of letting the machine swap itself to a halt."""
    avail = available_gb()
    if avail is not None and avail < min_gb:
        raise NotEnoughMemory(f"free memory fell to {avail:.1f} GB during restoration; stopped to keep the "
                              "computer responsive. Close other applications and run again.")
