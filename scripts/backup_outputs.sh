#!/bin/sh
# Copy the project's outputs that exist only on this laptop (results/, kaggle_runs/, docs/, the split files and
# label tables) to a backup folder, normally one inside Google Drive for desktop, which uploads it.
#
# The destination comes from MEDINTERP_BACKUP_DIR or the gitignored .local/backup_dir, so that no personal path is written into the repository.
# Nothing is deleted at the destination: a file removed here stays in the backup, so a mistake on the laptop
# cannot spread to the copy. Left out: the Stanford AIMI data (data/chexpert_plus; the agreement bars passing it
# on, see docs/framing.md section 10), the UMedPT weights (data/weights; downloadable again and checked by
# SHA-256), the activation upload staging folder (kaggle_runs/_upload; hard links to the harvest, which is kept on
# Kaggle) and Python caches.
#
# Run:  MEDINTERP_BACKUP_DIR=... sh scripts/backup_outputs.sh [--dry-run]
set -eu

repo=$(cd "$(dirname "$0")/.." && pwd)
# Fallback: the gitignored file .local/backup_dir (one line, the folder), so the setting survives new shells.
if [ -z "${MEDINTERP_BACKUP_DIR:-}" ] && [ -f "$repo/.local/backup_dir" ]; then
    MEDINTERP_BACKUP_DIR=$(head -n 1 "$repo/.local/backup_dir")
fi
if [ -z "${MEDINTERP_BACKUP_DIR:-}" ]; then
    echo "set MEDINTERP_BACKUP_DIR or write the folder into .local/backup_dir" >&2
    exit 1
fi
if [ ! -d "$(dirname "$MEDINTERP_BACKUP_DIR")" ]; then
    echo "parent of $MEDINTERP_BACKUP_DIR does not exist (is Google Drive running?)" >&2
    exit 1
fi

mkdir -p "$MEDINTERP_BACKUP_DIR"
# -a keeps times so that later runs copy only what changed; --relative keeps the repository layout.
cd "$repo"
rsync -a --relative ${1:+"$1"} --itemize-changes \
    --exclude '__pycache__/' --exclude '*.pyc' --exclude '.DS_Store' \
    --exclude 'kaggle_runs/_upload/' \
    results/ kaggle_runs/ docs/ data/splits/ data/labels/ \
    "$MEDINTERP_BACKUP_DIR/" | grep -c '^[>c]' | sed 's/^/files copied or updated: /'
