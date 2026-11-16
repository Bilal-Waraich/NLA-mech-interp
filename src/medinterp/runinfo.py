"""Run bookkeeping shared by the entry points: ``config.json`` with settings, seeds, code revision, the
split file's SHA-256 and package versions (build spec section 2)."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

PACKAGES = ("numpy", "torch", "scipy", "scikit-learn", "pandas", "safetensors")


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_revision(repo: Path | str) -> str:
    """HEAD of the repository, with ``+dirty`` if tracked files differ from it; "unknown" without git."""
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo,
                               capture_output=True, text=True, check=True).stdout.strip()
        return rev + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def package_versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def write_config(out_dir: Path, settings: dict, repo: Path, split_file: Path | None = None) -> dict:
    """Write ``out_dir/config.json`` and return its content."""
    cfg = {"settings": settings, "git_revision": git_revision(repo), "packages": package_versions()}
    if split_file is not None:
        cfg["split_file"] = str(split_file)
        cfg["split_sha256"] = sha256_file(split_file)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    return cfg
