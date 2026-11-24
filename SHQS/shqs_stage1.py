"""STAGE 1 - SHQS (uses shqs_opencv.py UNCHANGED).

Scores every image, prints the table, and writes the hand-off package:
  <folder>_run/01_shqs/manifest.csv      <- THE hand-off file (one row per image)
  <folder>_run/01_shqs/README.txt        <- guide for the next stages
  <folder>_run/01_shqs/stage1_info.json  <- counts, thresholds, run-order
  <folder>_run/01_shqs/no|soft|strong/   <- copies of the originals, split by level
Nothing is enhanced here.

Run:  python shqs_stage1.py "my_folder"      (or set FOLDER below)
"""

import csv, json, shutil, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import shqs_opencv as S  # the user's code, untouched
from handoff import COLUMNS, ROUTE, run_root

FOLDER = "test1"  # <-- change, or pass as argument
RECURSIVE = True


def uid(rel):  # unique, filesystem-safe id for an image
    return "__".join(rel.with_suffix("").parts)


def main():
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else FOLDER)
    folder = S.find_folder(str(folder)).resolve()
    root = run_root(folder) / "01_shqs"
    for lv in ("no", "soft", "strong"):
        (root / lv).mkdir(parents=True, exist_ok=True)

    it = folder.rglob("*") if RECURSIVE else folder.glob("*")
    files = sorted(
        p
        for p in it
        if p.suffix.lower() in S.IMAGE_EXTS and p.is_file() and "_run" not in p.parts
    )
    if not files:
        raise SystemExit(f"No images in {folder}")

    rows, skipped = [], []
    print(f"\n{'image':<40}{'SHQS':>7}  {'level':<7}{'weakest':<8}route")
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            skipped.append(str(p))
            continue
        rel = p.relative_to(folder)
        q, mu, found, _ = S.shqs_score(img)
        lvl = S.level_of(q)
        weakest = min(mu, key=mu.get)
        i = uid(rel)
        split = root / lvl.lower() / (i + p.suffix.lower())
        shutil.copy2(p, split)
        rows.append(
            {
                "id": i,
                "rel_path": rel.as_posix(),
                "image_path": str(p),
                "split_path": str(split),
                "shqs": f"{q:.2f}",
                "level": lvl,
                "route": ROUTE[lvl],
                "weakest": weakest,
                **{k: f"{mu[k]:.3f}" for k in "SRFNCL"},
                "height": img.shape[0],
                "width": img.shape[1],
            }
        )
        print(f"{rel.as_posix()[:39]:<40}{q:7.2f}  {lvl:<7}{weakest:<8}{ROUTE[lvl]}")

    with open(root / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    cnt = Counter(r["level"] for r in rows)
    info = {
        "stage": "01_shqs",
        "input_folder": str(folder),
        "total": len(rows),
        "counts": {k: cnt.get(k, 0) for k in ("NO", "SOFT", "STRONG")},
        "thresholds": {"NO_if_shqs_>=": S.SOFT_THR, "STRONG_if_shqs_<": S.STRONG_THR},
        "skipped_unreadable": skipped,
        "shqs_version": S.VERSION,
        "run_order": [
            "01 shqs_stage1.py",
            "02 stage2_clahe.py (SOFT only)",
            "03 CodeFormer teammate (STRONG only) -> 03_codeformer/",
            "04 stage_merge.py -> 04_ready_for_matching/",
        ],
    }
    (root / "stage1_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (root / "README.txt").write_text(
        f"""HAND-OFF GUIDE (written by stage 1)
==================================
manifest.csv has ONE ROW PER IMAGE. Each stage filters it by `level` and touches only its rows.

  level   route        who handles it                 input file to read
  NO      matching     nobody (already good)          image_path / split_path
  SOFT    clahe        stage2_clahe.py                split_path   (folder soft/)
  STRONG  codeformer   teammate's CodeFormer stage    split_path   (folder strong/)

Counts this run: NO={cnt.get('NO',0)}  SOFT={cnt.get('SOFT',0)}  STRONG={cnt.get('STRONG',0)}  (total {len(rows)})

RULES
 1. Process ONLY rows whose level matches your stage. Never output the others.
 2. Save your output as <id>.<any image ext> (the `id` column is the file stem) so merge can find it.
 3. CLAHE output goes to ../02_clahe/ ; CodeFormer output goes to ../03_codeformer/.
 4. Do not edit manifest.csv. Do not rename ids.

CodeFormer teammate (python):
    from handoff import read_manifest, select
    rows = select(read_manifest(r"{root / 'manifest.csv'}"), "STRONG")
    for r in rows: run_codeformer(r["split_path"], save_as=r["id"] + ".png")
(or simply point CodeFormer at the folder: {root / 'strong'})

Columns: id, rel_path, image_path (original), split_path (copy for this stage), shqs (0-100),
level, route, weakest (lowest factor: S sharp, R resolution, F face, N noise, C contrast,
L light), S R F N C L (0-1 scores), height, width.
Tip: weakest=R means the face is too small - CLAHE cannot add resolution.
""",
        encoding="utf-8",
    )
    print(
        f"\nNO={cnt.get('NO',0)}  SOFT={cnt.get('SOFT',0)}  STRONG={cnt.get('STRONG',0)}"
        f"  (skipped {len(skipped)})\nHand-off written to: {root}\n"
    )


if __name__ == "__main__":
    main()
