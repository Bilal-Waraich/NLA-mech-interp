"""Download the output of finished Kaggle runs to ``results/<study>/kaggle/<slug>/``, then refresh the run index
(``scripts/run_index.py``) and the backup (``scripts/backup_outputs.sh``), so that every download leaves the index
and the backup up to date (build spec section 6).

The account that owns a run is found by asking each project account in turn, so no account name is needed on the
command line. A run that is still running or queued is skipped with a message. Runs that Kaggle stopped at its
12 h limit (CANCEL_ACKNOWLEDGED) keep their saved files and are downloaded like finished ones.

Run:  ~/.venvs/nla-mech-interp/bin/python scripts/kaggle_fetch.py --study sweep_v1 <slug> [<slug> ...]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kaggle_sync import kaggle, project_accounts  # noqa: E402  (sibling script: accounts and tokens)

REPO = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
DOWNLOADABLE = {"complete", "error", "cancel_acknowledged"}


def owner_and_status(slug: str) -> tuple[str, str] | None:
    """The first project account that can see the run, and the run's status there."""
    for account in project_accounts():
        res = kaggle(account, "kernels", "status", f"{account}/{slug}", check=False)
        found = re.search(r"KernelWorkerStatus\.(\w+)", res.stdout)
        if found:
            return account, found.group(1).lower()
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("slugs", nargs="+")
    ap.add_argument("--study", required=True, help="folder under results/, e.g. sweep_v1")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    fetched = 0
    for slug in args.slugs:
        found = owner_and_status(slug)
        if found is None:
            print(f"{slug}: not found on any project account")
            continue
        account, status = found
        if status not in DOWNLOADABLE:
            print(f"{slug}: {status}, not downloaded")
            continue
        dest = REPO / "results" / args.study / "kaggle" / slug
        dest.mkdir(parents=True, exist_ok=True)
        res = kaggle(account, "kernels", "output", f"{account}/{slug}", "-p", str(dest), check=False)
        if res.returncode != 0:
            print(f"{slug}: download failed")     # stderr is not echoed: it would show the account name
            continue
        n_files = sum(1 for f in dest.rglob("*") if f.is_file())
        print(f"{slug}: {status}, {n_files} files in {dest.relative_to(REPO)}")
        fetched += 1
    if not fetched:
        return
    subprocess.run([PYTHON, str(REPO / "scripts/run_index.py")], check=True)
    if not args.no_backup:
        subprocess.run(["sh", str(REPO / "scripts/backup_outputs.sh")], check=True)


if __name__ == "__main__":
    main()
