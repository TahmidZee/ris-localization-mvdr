#!/usr/bin/env python3
"""
Safe cleanup utility for this repo (dry-run by default).

Why:
  - Clear python caches, transient logs, and optionally old checkpoints/HPO artifacts
  - WITHOUT accidentally deleting "best" models or the Optuna DB unless explicitly requested

Usage:
  python cleanup_artifacts.py --dry-run
  python cleanup_artifacts.py --yes --pycache --logs
  python cleanup_artifacts.py --yes --runs --keep-best
  python cleanup_artifacts.py --yes --hpo --wipe-hpo   # DANGEROUS: deletes hpo.db/journal

By default, this script:
  - does NOT delete checkpoints
  - does NOT delete HPO DB
  - does NOT delete datasets
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List


def _iter_dirs(root: Path, names: Iterable[str]) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_dir() and p.name in names:
            yield p


def _iter_files(root: Path, globs: Iterable[str]) -> Iterable[Path]:
    for g in globs:
        for p in root.rglob(g):
            if p.is_file():
                yield p


def _rm_path(p: Path, dry_run: bool) -> int:
    try:
        if dry_run:
            print(f"[DRY] rm -rf {p}")
            return 1
        if p.is_dir():
            # recursive delete
            for child in sorted(p.rglob("*"), reverse=True):
                if child.is_file() or child.is_symlink():
                    child.unlink(missing_ok=True)
                elif child.is_dir():
                    try:
                        child.rmdir()
                    except OSError:
                        pass
            p.rmdir()
        else:
            p.unlink(missing_ok=True)
        print(f"[DEL] {p}")
        return 1
    except Exception as e:
        print(f"[SKIP] {p} ({type(e).__name__}: {e})")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="Repo root (default: current directory)")
    ap.add_argument("--yes", action="store_true", help="Actually delete (otherwise dry-run)")
    ap.add_argument("--pycache", action="store_true", help="Delete __pycache__/ and *.pyc/*.pyo")
    ap.add_argument("--logs", action="store_true", help="Delete *.log/*.out/*.err under results dirs")
    ap.add_argument("--runs", action="store_true", help="Delete old checkpoints under cfg.RESULTS_DIR/checkpoints (keeps best unless --no-keep-best)")
    ap.add_argument("--no-keep-best", action="store_true", help="If set, also delete best checkpoints (DANGEROUS)")
    ap.add_argument("--hpo", action="store_true", help="Delete transient HPO logs (NOT the DB unless --wipe-hpo)")
    ap.add_argument("--wipe-hpo", action="store_true", help="Delete hpo.db and journals too (DANGEROUS)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    dry_run = not bool(args.yes)

    deleted = 0

    # 1) python caches
    if args.pycache:
        for d in _iter_dirs(root, names=["__pycache__", ".pytest_cache"]):
            deleted += _rm_path(d, dry_run=dry_run)
        for f in _iter_files(root, globs=["*.pyc", "*.pyo"]):
            deleted += _rm_path(f, dry_run=dry_run)

    # 2) results/logs/checkpoints paths (best-effort; doesn't require importing cfg)
    # Heuristic: these dirs exist in this repo.
    results_dirs: List[Path] = []
    for cand in ["results_final_L16_12x12", "results_final"]:
        p = root / cand
        if p.exists() and p.is_dir():
            results_dirs.append(p)

    if args.logs:
        for rd in results_dirs:
            for f in _iter_files(rd, globs=["*.log", "*.out", "*.err"]):
                deleted += _rm_path(f, dry_run=dry_run)

    if args.hpo:
        for rd in results_dirs:
            hpo_dir = rd / "hpo"
            if not hpo_dir.exists():
                continue
            # delete timestamped logs/csv (keep best.json by default)
            for f in _iter_files(hpo_dir, globs=["hpo_*.log", "*_trials.csv", "*.journal"]):
                # journals are safe unless user wants to resume; keep unless wipe requested
                if f.suffix == ".journal" and not args.wipe_hpo:
                    continue
                deleted += _rm_path(f, dry_run=dry_run)
            if args.wipe_hpo:
                for f in _iter_files(hpo_dir, globs=["hpo.db", "hpo.db-*"]):
                    deleted += _rm_path(f, dry_run=dry_run)

    if args.runs:
        keep_best = not args.no_keep_best
        keep_names = {"best.pt", "best.ckpt", "swa.pt", "run_config.json"}
        for rd in results_dirs:
            ckpt_dir = rd / "checkpoints"
            if not ckpt_dir.exists():
                continue
            for f in ckpt_dir.glob("*"):
                if f.is_dir():
                    continue
                if keep_best and f.name in keep_names:
                    continue
                deleted += _rm_path(f, dry_run=dry_run)

    if deleted == 0:
        print("Nothing to delete (or paths not found).")
    else:
        print(f"Planned deletions: {deleted}" if dry_run else f"Deleted: {deleted}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


