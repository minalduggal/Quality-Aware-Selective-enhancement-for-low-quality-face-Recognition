"""handoff.py - shared contract between pipeline stages.

Every stage reads 01_shqs/manifest.csv through this file. Teammates (CodeFormer, matching)
only need:   from handoff import read_manifest, select
"""

import csv
from pathlib import Path

ROUTE = {"NO": "matching", "SOFT": "clahe", "STRONG": "codeformer"}
COLUMNS = [
    "id",
    "rel_path",
    "image_path",
    "split_path",
    "shqs",
    "level",
    "route",
    "weakest",
    "S",
    "R",
    "F",
    "N",
    "C",
    "L",
    "height",
    "width",
]


def run_root(folder):
    """<folder>_run/ next to the input folder."""
    p = Path(folder).resolve()
    return p.parent / (p.name + "_run")


def read_manifest(path):
    """Return manifest rows as dicts (numbers converted)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("shqs", "S", "R", "F", "N", "C", "L"):
            r[k] = float(r[k]) if r.get(k) not in (None, "") else None
        for k in ("height", "width"):
            r[k] = int(r[k]) if r.get(k) else None
    return rows


def select(rows, level):
    """Only the rows of one level: 'NO', 'SOFT' or 'STRONG'. Nothing else leaks through."""
    return [r for r in rows if r["level"] == level]


def find_output(folder, image_id):
    """Find a stage's output for image `id` (matched by file stem; ext may differ)."""
    folder = Path(folder)
    if not folder.is_dir():
        return None
    for suffix in ("", "_out", "_restored", "_00"):
        for p in folder.rglob(image_id + suffix + ".*"):
            if p.is_file():
                return p
    return None
