"""Files the skill cannot run without must be tracked by git (a global gitignore once dropped uv.lock)."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["skills/make-normal-voice/uv.lock", "skills/make-normal-voice/pyproject.toml", "skills/make-normal-voice/SKILL.md",
            "skills/make-normal-voice/scripts/make-normal-voice", ".claude-plugin/plugin.json", ".claude-plugin/marketplace.json"]


def test_required_files_are_tracked():
    out = subprocess.run(["git", "ls-files", *REQUIRED], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:  # not a git checkout (e.g. a copied skill folder): nothing to check
        return
    tracked = set(out.stdout.split())
    assert tracked == set(REQUIRED), set(REQUIRED) - tracked
