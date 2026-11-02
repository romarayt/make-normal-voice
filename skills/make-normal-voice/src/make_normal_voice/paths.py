"""Where make-normal-voice keeps its environment, models and caches.

Everything lives in one per-user data folder so that nothing needs sudo and nothing lands in a
temporary directory that the OS may wipe (macOS cleans /private/tmp every few days).
The launcher script computes the same folder for the Python environment; keep the two in sync.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from platformdirs import user_data_dir

APP = "make-normal-voice"


def home() -> Path:
    env = os.environ.get("MAKE_NORMAL_VOICE_HOME")
    return Path(env).expanduser() if env else Path(user_data_dir(APP, appauthor=False))


def models_dir() -> Path:
    return home() / "models"


def cache_dir() -> Path:
    return home() / "cache"


def free_bytes(path: Path) -> int:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free
