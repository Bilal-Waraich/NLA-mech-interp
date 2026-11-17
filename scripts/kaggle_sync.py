"""Upload the project's private Kaggle datasets to every project account, from one local source (build
spec section 6; second account: framing decision log, 2026-11-16).

Datasets (slug the same on every account, always private):
  src    medinterp-src           src/, scripts/, configs/, pyproject.toml as one archive (no data)
  labels medinterp-labels        data/labels/, data/splits/
  acts   medinterp-umedpt-acts   archived UMedPT activations and the Stage 0.6 seeds (UMedPT-derived)

The first call per account creates the dataset (private is Kaggle's default; ``-u`` is never passed),
later calls add a version. ``--check`` lists each dataset's files and sizes on every account and fails
if they differ. Tokens are read from ``~/.kaggle`` and handed to the Kaggle tool through its environment;
they are never printed.

Run:  ~/.venvs/nla-mech-interp/bin/python scripts/kaggle_sync.py src -m "sweep code"
      ~/.venvs/nla-mech-interp/bin/python scripts/kaggle_sync.py --check
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KAGGLE = Path.home() / ".venvs/kaggle-cli/bin/kaggle"
ACCOUNTS_ENV = "MEDINTERP_KAGGLE_ACCOUNTS"
SLUGS = {"src": "medinterp-src", "labels": "medinterp-labels", "acts": "medinterp-umedpt-acts"}
UPLOAD = REPO / "kaggle_runs/_upload"
SRC_CONTENT = ("src", "scripts", "configs", "pyproject.toml")


def project_accounts() -> tuple[str, ...]:
    """Kaggle usernames of the project, default account first.

    Kept out of the public repository: read from the environment variable MEDINTERP_KAGGLE_ACCOUNTS
    (comma-separated). The default account authenticates with ~/.kaggle/access_token, every other one
    with ~/.kaggle/accounts/<name>.token.

    Raises SystemExit if the variable is unset or names no account.
    """
    names = tuple(n.strip() for n in os.environ.get(ACCOUNTS_ENV, "").split(",") if n.strip())
    if not names:
        sys.exit(f"no Kaggle accounts configured: set {ACCOUNTS_ENV} (comma-separated, default account first)")
    return names


def kaggle_env(account: str) -> dict:
    """Environment for one Kaggle call as ``account``; the default account uses ~/.kaggle/access_token."""
    env = dict(os.environ)
    env.pop("KAGGLE_API_TOKEN", None)
    token_file = Path.home() / ".kaggle/accounts" / f"{account}.token"
    if account != project_accounts()[0]:
        env["KAGGLE_API_TOKEN"] = token_file.read_text().strip()
    return env


def kaggle(account: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run([str(KAGGLE), *args], env=kaggle_env(account), capture_output=True, text=True,
                          check=check)


def build_src(folder: Path) -> None:
    """One archive with a fixed name: run.py unpacks it, whatever Kaggle does with folders."""
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for item in SRC_CONTENT:
            src = REPO / item
            if src.is_dir():
                shutil.copytree(src, stage / item, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            else:
                shutil.copy2(src, stage / item)
        shutil.make_archive(str(folder / "medinterp_src"), "gztar", stage)


def push(name: str, account: str, message: str) -> None:
    slug = SLUGS[name]
    with tempfile.TemporaryDirectory() as tmp:
        if name == "src":
            folder = Path(tmp) / slug
            folder.mkdir()
            build_src(folder)
        else:
            folder = UPLOAD / slug          # prepared once (hard links to the local files)
            if not folder.exists():
                sys.exit(f"{folder} missing")
        (folder / "dataset-metadata.json").write_text(json.dumps(
            {"title": slug, "id": f"{account}/{slug}", "licenses": [{"name": "other"}]}))
        ref = f"{account}/{slug}"
        status = kaggle(account, "datasets", "status", ref, check=False)
        exists = status.returncode == 0 and "ready" in status.stdout.lower()
        args = (["datasets", "version", "-p", str(folder), "-m", message, "-r", "zip"] if exists
                else ["datasets", "create", "-p", str(folder), "-r", "zip"])
        res = subprocess.run([str(KAGGLE), *args], env=kaggle_env(account))
        if res.returncode != 0:
            sys.exit(f"upload of {ref} failed")
        print(f"{ref}: {'new version' if exists else 'created'}", flush=True)


def listing(account: str, slug: str) -> list[tuple[str, str]] | None:
    res = kaggle(account, "datasets", "files", f"{account}/{slug}", "--csv", check=False)
    if res.returncode != 0:
        return None
    rows = [line.split(",") for line in res.stdout.strip().splitlines()[1:]]
    return sorted((r[0], r[1]) for r in rows if r)


def check() -> bool:
    ok = True
    for name, slug in SLUGS.items():
        lists = {a: listing(a, slug) for a in project_accounts()}
        same = len({json.dumps(v) for v in lists.values()}) == 1 and None not in lists.values()
        ok &= same
        print(f"{slug}: {'in sync' if same else 'DIFFERENT'} " +
              ", ".join(f"{a}: {'missing' if v is None else f'{len(v)} files'}" for a, v in lists.items()))
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", help=f"which datasets to upload: {list(SLUGS)}")
    ap.add_argument("-m", "--message", default="update")
    accounts = project_accounts()
    ap.add_argument("--accounts", nargs="+", default=list(accounts), choices=accounts)
    ap.add_argument("--check", action="store_true", help="compare file lists across accounts")
    args = ap.parse_args()
    bad = set(args.datasets) - set(SLUGS)
    if bad:
        ap.error(f"unknown datasets {sorted(bad)}")
    for name in args.datasets:
        for account in args.accounts:
            push(name, account, args.message)
    if args.check and not check():
        sys.exit(1)


if __name__ == "__main__":
    main()
