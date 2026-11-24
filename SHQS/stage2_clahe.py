"""STAGE 2 - CLAHE for SOFT images only.

Reads 01_shqs/manifest.csv, takes ONLY rows with level == SOFT, writes ONLY those to
<folder>_run/02_clahe/ (+ clahe_manifest.csv). NO and STRONG images are never read or written.

Run:  python stage2_clahe.py "my_folder"   (same folder name given to stage 1)
"""

import csv, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2, numpy as np
from handoff import read_manifest, select, run_root
import shqs_stage1 as s1

CLAHE_CLIP, CLAHE_MIN_TILE_PX, CLAHE_MAX_TILES = 2.0, 16, 8


def clahe_enhance(
    bgr, clip=CLAHE_CLIP, min_tile_px=CLAHE_MIN_TILE_PX, max_tiles=CLAHE_MAX_TILES
):
    """CLAHE on LAB-L only (colour kept); tile grid adapts to image size."""
    h, w = bgr.shape[:2]
    tiles = int(np.clip(min(h, w) // min_tile_px, 2, max_tiles))
    l, a, b = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB))
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def main():
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else s1.FOLDER).resolve()
    run = run_root(folder)
    manifest = run / "01_shqs" / "manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"manifest not found: {manifest}\nRun shqs_stage1.py first.")
    rows = select(read_manifest(manifest), "SOFT")
    out = run / "02_clahe"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*"):  # fresh run -> no stale files
        if old.is_file():
            old.unlink()
    done = []
    for r in rows:
        img = cv2.imread(r["split_path"])
        if img is None:
            print("unreadable:", r["split_path"])
            continue
        cv2.imwrite(str(out / (r["id"] + ".png")), clahe_enhance(img))
        done.append(
            {
                "id": r["id"],
                "input": r["split_path"],
                "output": str(out / (r["id"] + ".png")),
            }
        )
    with open(out / "clahe_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "input", "output"])
        w.writeheader()
        w.writerows(done)
    print(f"CLAHE: {len(done)} SOFT images enhanced -> {out}")


if __name__ == "__main__":
    main()
