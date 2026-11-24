"""
shqs_opencv.py  -  NO / SOFT / STRONG enhancement decision.  OpenCV + NumPy only.
No model downloads (no ArcFace, no RetinaFace), no weighted sum.

HOW TO USE (VS Code):
  1. One-time:  pip install opencv-python numpy
  2. Put this file and your image folder in the SAME directory.
  3. Set FOLDER below and press Run.

THE SCORE (fuzzy memberships multiplied together, no weights)
  Each quality factor becomes a membership value in [0, 1]:
        mu = clip( (x - bad) / (good - bad), 0, 1 )      1 = fine, 0 = unusable
  Factors:  S sharpness (blur + low resolution; noise-compensated, face interior, relative to the
                       face's own contrast so dark faces are not mistaken for blurry ones)
            R resolution (native size of the file; a 20x25 px crop upscaled for analysis looks
                          artificially sharp, so size is scored on its own)
            F fidelity   (is a real, detectable face present? OpenCV Haar frontal/profile check)
            N noise      (Immerkaer estimate)
            C contrast   (percentile range)
            L illumination (distance of mean brightness from mid-grey)

        SHQS = 100 * S * R * F * N * C * L

  A product is not a weighted sum: every factor can veto. If any one factor is near 0 the
  whole score is near 0, and a very good factor cannot hide a very bad one. Importance is
  expressed through how strict each ramp is (S and F are the strictest), not through weights.

  Level:  NO      SHQS >= SOFT_THR
          SOFT    STRONG_THR <= SHQS < SOFT_THR
          STRONG  SHQS < STRONG_THR   (includes "no face found")
"""

import csv
import math
import sys
import warnings
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore")

# =====================================================================================
#  EDIT THESE
# =====================================================================================
FOLDER = "crops/raw"  # folder with your face images (next to this file, or a full path)
SOFT_THR = 75.0  # SHQS below this -> at least SOFT   (clear pictures are above it)
STRONG_THR = 35.0  # SHQS below this -> STRONG          (not recognisable)
# SAVE_ENHANCED = False  # write enhanced copies to <FOLDER>_enhanced/soft and /strong
OUT_CSV = "results_opencv.csv"

# membership ramps: (bad, good). x <= bad -> 0, x >= good -> 1 (works in either direction)
RAMPS = {
    "sharp_rel": (
        0.05,
        0.22,
    ),  # detail energy relative to the face's own contrast (blur / low-res)
    "sharp_abs": (
        0.5,
        3.0,
    ),  # loose absolute floor: stops a flat, featureless crop passing
    "res": (
        10.0,
        48.0,
    ),  # NATIVE size (shorter side, pixels) of the file: tiny crops are unrecognisable
    "noise": (20.0, 5.0),  # Immerkaer sigma in grey levels, lower is better
    "contrast": (0.10, 0.35),
    "illum": (0.48, 0.30),  # |mean brightness - 0.5|, lower is better
}
NO_FACE_F = 0.05  # F when no face is found
# =====================================================================================

VERSION = "v5 (S*R*F*N*C*L, native-size factor)"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FACE_INTERIOR = 0.45
BAND_SIGMAS = (1.5, 3.5)
_BAND_GAIN = []
_CASCADES = None


def ramp(x, key):
    bad, good = RAMPS[key]
    return float(np.clip((x - bad) / (good - bad), 0.0, 1.0))


# ---------------------------------------------------------------- measurements ----
def _central(g, f):
    n = g.shape[0]
    m = int(n * (1 - f) / 2)
    return g[m : n - m, m : n - m]


def _band(g):
    return cv2.GaussianBlur(g, (0, 0), BAND_SIGMAS[0]) - cv2.GaussianBlur(
        g, (0, 0), BAND_SIGMAS[1]
    )


def _immerkaer(g):
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    h, w = g.shape
    return (
        math.sqrt(math.pi / 2)
        / (6 * (w - 2) * (h - 2))
        * float(np.abs(cv2.filter2D(g, -1, k)).sum())
    )


def measure(bgr):
    """Raw measurements of one face crop."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    interp = (
        cv2.INTER_AREA if min(g.shape) >= 128 else cv2.INTER_CUBIC
    )  # small crops get blurry
    g = cv2.resize(g, (128, 128), interpolation=interp)

    noise = _immerkaer(g)
    if not _BAND_GAIN:  # band-pass gain on unit white noise
        wn = np.random.RandomState(0).normal(0, 1, (256, 256)).astype(np.float32)
        _BAND_GAIN.append(float(_central(_band(wn), 0.6).std()))
    e = float(
        _central(_band(g), FACE_INTERIOR).std()
    )  # detail energy, face interior only
    sharp = math.sqrt(
        max(e * e - (_BAND_GAIN[0] * noise) ** 2, 0.0)
    )  # minus the grain contribution
    sharp_rel = sharp / (
        float(_central(g, FACE_INTERIOR).std()) + 1e-6
    )  # relative to the face's own contrast,
    # so dark / washed-out faces are not "blurry"

    sm = cv2.GaussianBlur(g, (0, 0), 1.5)
    p2, p98 = np.percentile(sm, [2, 98])
    clipped = float(((sm <= 5) | (sm >= 250)).mean())
    contrast = ((p98 - p2) / 255.0) * (1 - clipped)
    illum = abs(float(g.mean()) / 255.0 - 0.5)
    return {
        "sharp": sharp,
        "sharp_rel": sharp_rel,
        "noise": noise,
        "contrast": contrast,
        "illum": illum,
    }


def face_found(bgr):
    """Fidelity check with OpenCV's bundled Haar cascades (frontal + profile, both sides).
    Returns True/False, or None if the cascade files are not available."""
    global _CASCADES
    if _CASCADES is None:
        try:
            names = [
                "haarcascade_frontalface_default.xml",
                "haarcascade_profileface.xml",
            ]
            _CASCADES = [
                cv2.CascadeClassifier(cv2.data.haarcascades + n) for n in names
            ]
            if any(c.empty() for c in _CASCADES):
                raise RuntimeError
        except Exception:
            print(
                "WARNING: OpenCV Haar cascade files not found - face check skipped (F = 1). "
                "Try: pip install opencv-python"
            )
            _CASCADES = []
    if not _CASCADES:
        return None
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    s = 160.0 / max(g.shape)
    g = cv2.resize(
        g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    )
    p = int(0.3 * max(g.shape))
    g = cv2.equalizeHist(cv2.copyMakeBorder(g, p, p, p, p, cv2.BORDER_REFLECT))
    front, prof = _CASCADES
    return (
        len(front.detectMultiScale(g, 1.1, 3, minSize=(40, 40))) > 0
        or len(prof.detectMultiScale(g, 1.1, 3, minSize=(40, 40))) > 0
        or len(prof.detectMultiScale(cv2.flip(g, 1), 1.1, 3, minSize=(40, 40))) > 0
    )


# ----------------------------------------------------------------------- score ----
def shqs_score(bgr):
    m = measure(bgr)
    found = face_found(bgr)
    mu = {
        "S": min(ramp(m["sharp_rel"], "sharp_rel"), ramp(m["sharp"], "sharp_abs")),
        "R": ramp(float(min(bgr.shape[:2])), "res"),
        "F": 1.0 if found is None else (1.0 if found else NO_FACE_F),
        "N": ramp(m["noise"], "noise"),
        "C": ramp(m["contrast"], "contrast"),
        "L": ramp(m["illum"], "illum"),
    }
    q = 100.0 * float(np.prod(list(mu.values())))
    return q, mu, found, m


def level_of(q):
    if q >= SOFT_THR:
        return "NO"
    return "STRONG" if q < STRONG_THR else "SOFT"


# # ----------------------------------------------------------------- enhancement ----
# def _clahe_lab(bgr, clip, tiles=8):
#     lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(l)
#     return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# def soft_enhance(bgr):
#     """Light: CLAHE (clip 2.0) on the lightness channel."""
#     return _clahe_lab(bgr, 2.0)


# def strong_enhance(bgr):
#     """Heavy: denoise -> CLAHE (clip 3.0) -> unsharp mask."""
#     den = cv2.fastNlMeansDenoisingColored(bgr, None, 5, 5, 7, 21)
#     eq = _clahe_lab(den, 3.0)
#     return cv2.addWeighted(eq, 1.8, cv2.GaussianBlur(eq, (0, 0), 1.5), -0.8, 0)


# ------------------------------------------------------------------------ main ----
def find_folder(name):
    p = Path(name)
    if p.is_dir():
        return p
    here = Path(__file__).resolve().parent / name
    if here.is_dir():
        return here
    raise SystemExit(
        f"\nFolder not found: '{name}'\n  looked in: {Path.cwd()}\n"
        f"        and: {Path(__file__).resolve().parent}\n"
        f"  Fix: set FOLDER at the top of this file."
    )


def main():
    root = find_folder(sys.argv[1] if len(sys.argv) > 1 else FOLDER)
    files = sorted(p for p in root.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise SystemExit(f"No images in {root.resolve()}")
    # out_root = root.parent / (root.name + "_enhanced")
    # if SAVE_ENHANCED:
    #     for sub in ("soft", "strong"):
    #         (out_root / sub).mkdir(parents=True, exist_ok=True)

    rows, skipped = [], []
    for p in files:
        im = cv2.imread(str(p))
        if im is None:
            skipped.append(p.name)
            continue
        q, mu, found, raw = shqs_score(im)
        level = level_of(q)
        weakest = min(mu, key=mu.get)
        # if SAVE_ENHANCED and level != "NO":
        #     cv2.imwrite(
        #         str(out_root / level.lower() / p.name),
        #         soft_enhance(im) if level == "SOFT" else strong_enhance(im),
        #     )
        rows.append(
            {
                "file": p.name,
                "shqs": round(q, 1),
                "level": level,
                "weakest": weakest if level != "NO" else "-",
                **{k: round(v, 2) for k, v in mu.items()},
                "size_hw": f"{im.shape[0]}x{im.shape[1]}",
                "face_found": found,
                "raw_sharp_rel": round(raw["sharp_rel"], 3),
                "raw_sharp": round(raw["sharp"], 2),
                "raw_noise": round(raw["noise"], 2),
                "raw_contrast": round(raw["contrast"], 3),
            }
        )

    out_csv = Path(__file__).resolve().parent / OUT_CSV
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    rows.sort(key=lambda r: r["shqs"])
    print(f"Script: {Path(__file__).resolve()}  |  version {VERSION}")
    print(f"Folder: {root.resolve()}")
    print(
        f"SHQS = 100 * S*R*F*N*C*L   |   NO >= {SOFT_THR:g}  |  STRONG < {STRONG_THR:g}  |  else SOFT\n"
    )
    print(
        f"{'file':<28}{'SHQS':>6}  {'level':<8}{'S':>5}{'R':>5}{'F':>5}{'N':>5}{'C':>5}{'L':>5}  {'weakest'}"
    )
    for r in rows:
        print(
            f"{r['file'][:27]:<28}{r['shqs']:>6}  {r['level']:<8}{r['S']:>5.2f}{r['R']:>5.2f}{r['F']:>5.2f}"
            f"{r['N']:>5.2f}{r['C']:>5.2f}{r['L']:>5.2f}  {r['weakest']}"
        )
    cnt = Counter(r["level"] for r in rows)
    print(f"\nNO: {cnt['NO']}  |  SOFT: {cnt['SOFT']}  |  STRONG: {cnt['STRONG']}")
    print(f"Results saved to {out_csv}")
    # if SAVE_ENHANCED:
    #     print(f"Enhanced images saved to {out_root}")
    if skipped:
        print(f"Unreadable, skipped: {skipped}")


if __name__ == "__main__":
    main()
