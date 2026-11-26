"""Write ``results/run_index.csv``: one row per Kaggle run of this project, so that every run, its inputs and its
downloaded outputs can be found from one table.

Sources: the generated run folders ``kaggle_runs/<name>/`` (``kernel-metadata.json``, ``run.py``), the live Kaggle
status of each run (``--no-status`` keeps the last known one), the downloads under ``results/`` and the notes files that mention
the run. Accounts are written as ``default`` / ``second`` (their order in ``MEDINTERP_KAGGLE_ACCOUNTS``), never by
name, because the second account's owner is not named in project files.

Run:  ~/.venvs/nla-mech-interp/bin/python scripts/run_index.py [--no-status]
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kaggle_sync import kaggle, project_accounts  # noqa: E402  (sibling script: accounts and tokens)

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "kaggle_runs"
RESULTS = REPO / "results"
NOTES = REPO / "docs/notes"
COLUMNS = ["slug", "account", "kind", "script", "args", "datasets", "kernel_sources", "pushed_utc", "status",
           "downloaded_to", "download_kb", "notes"]
# Early runs were downloaded into folders named differently from their slug.
DOWNLOAD_ALIASES = {"medinterp-bench-cpu": "kaggle_cpu", "medinterp-bench-t4": "kaggle_t4",
                    "medinterp-sweep-phase0-eval": "phase0-eval"}


def kaggle_slug(metadata_id: str) -> str:
    """Kaggle turns underscores of the pushed id into hyphens; the index uses the slug Kaggle shows."""
    return metadata_id.split("/", 1)[1].replace("_", "-")


def run_call(run_py: Path) -> tuple[str, str]:
    """The script and argument list that the generated ``run.py`` passes on (empty if not found)."""
    text = run_py.read_text() if run_py.exists() else ""
    script = re.search(r"^SCRIPT = (.*)$", text, re.M)
    args = re.search(r"^ARGS = (.*)$", text, re.M)
    return (ast.literal_eval(script.group(1)) if script else "",
            " ".join(ast.literal_eval(args.group(1))) if args else "")


def find_download(names: set[str]) -> Path | None:
    """The folder under ``results/`` that holds a run's downloaded output, matched by folder name."""
    for path in sorted(RESULTS.rglob("*")):
        if path.is_dir() and path.name in names:
            return path
    return None


def folder_kb(path: Path) -> int:
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e3)


def status_of(owner: str, slug: str) -> str:
    res = kaggle(owner, "kernels", "status", f"{owner}/{slug}", check=False)
    found = re.search(r"KernelWorkerStatus\.(\w+)", res.stdout)
    return found.group(1).lower() if found else "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-status", action="store_true", help="do not query Kaggle for each run's status")
    ap.add_argument("--out", type=Path, default=RESULTS / "run_index.csv")
    args = ap.parse_args()

    accounts = project_accounts()
    role = {name: ("default" if i == 0 else "second" if i == 1 else f"account{i}") for i, name in enumerate(accounts)}
    notes = {p.name: p.read_text() for p in sorted(NOTES.glob("*.md"))} if NOTES.exists() else {}
    # Without live queries the last known status is kept, so a quick refresh never blanks the column.
    known = {}
    if args.no_status and args.out.exists():
        with args.out.open() as fh:
            known = {r["slug"]: r.get("status", "") for r in csv.DictReader(fh)}
    rows = []
    for folder in sorted(p for p in RUNS.iterdir() if p.is_dir() and not p.name.startswith("_")):
        meta_file = folder / "kernel-metadata.json"
        if not meta_file.exists():
            continue
        meta = json.loads(meta_file.read_text())
        owner = meta["id"].split("/", 1)[0]
        slug = kaggle_slug(meta["id"])
        script, call_args = run_call(folder / "run.py")
        download = find_download({slug, folder.name, DOWNLOAD_ALIASES.get(slug, slug)})
        pushed = datetime.fromtimestamp(meta_file.stat().st_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M")
        strip = lambda refs: " ".join(r.split("/", 1)[1] for r in refs)  # owner prefix would name the account
        rows.append({
            "slug": slug, "account": role.get(owner, "other"),
            "kind": "gpu" if str(meta.get("enable_gpu")).lower() == "true" else "cpu",
            "script": script, "args": call_args,
            "datasets": strip(meta.get("dataset_sources", [])), "kernel_sources": strip(meta.get("kernel_sources", [])),
            "pushed_utc": pushed,
            "status": known.get(slug, "") if args.no_status or owner not in role else status_of(owner, slug),
            "downloaded_to": str(download.relative_to(REPO)) if download else "",
            "download_kb": folder_kb(download) if download else "",
            "notes": " ".join(n for n, text in notes.items() if slug in text or folder.name in text),
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    missing = [r["slug"] for r in rows if r["status"] in ("complete", "cancel_acknowledged") and not r["downloaded_to"]]
    print(f"{len(rows)} runs -> {args.out.relative_to(REPO)}; finished but not downloaded: {len(missing)}")
    for slug in missing:
        print(f"  {slug}")


if __name__ == "__main__":
    main()
