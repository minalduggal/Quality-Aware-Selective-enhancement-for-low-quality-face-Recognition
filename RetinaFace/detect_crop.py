"""
detect_crop.py - RetinaFace stage of the pipeline
Goal  : find faces and save one crop per face. Image quality is NOT judged here - SHQS does that.
        The detector score is used ONLY to tell "complete face" from "partial / not a real face"
        (in our tests, low-score crops were often just an ear, hair or a patch of skin).
Input : images in IMG_DIR
Output: crops/raw       faces with score >= KEEP_CONF  -> SHQS reads this
        crops/low_conf  uncertain detections (kept for inspection, NOT sent to SHQS)
        crops/meta      one JSON per crop: detector score, box, landmarks
"""

import os
import glob
import json

import cv2
from retinaface import RetinaFace

IMG_DIR = "images"
DET_THRESHOLD = 0.2  # RetinaFace's own cutoff: kept low so that we SEE every candidate
KEEP_CONF = 0.8  # score >= this -> crops/raw (goes to SHQS); below -> crops/low_conf
MARGIN = 0.10  # 10% extra around the box on each side

os.makedirs("crops/raw", exist_ok=True)  # -> goes to SHQS
os.makedirs("crops/low_conf", exist_ok=True)  # uncertain detections, not scored
os.makedirs("crops/meta", exist_ok=True)  # box, score, landmarks


def lm_list(lm):
    """Landmarks as 5 points [left_eye, right_eye, nose, mouth_left, mouth_right],
    sorted by x so the naming of left/right can't flip them.
    Values are converted to plain Python floats so json can save them."""
    eyes = sorted([lm["left_eye"], lm["right_eye"]], key=lambda p: p[0])
    mouth = sorted([lm["mouth_left"], lm["mouth_right"]], key=lambda p: p[0])
    pts = [eyes[0], eyes[1], lm["nose"], mouth[0], mouth[1]]
    return [[float(x), float(y)] for x, y in pts]


# # All images in IMG_DIR (to test a single image instead, use: paths = ["images/test3.png"])
# paths = sorted(
#     {
#         p
#         for ext in ("png", "jpg", "jpeg", "bmp")
#         for p in glob.glob(f"{IMG_DIR}/*.{ext}")
#     }
# )

# if not paths:
#     print(f"No images found in '{IMG_DIR}/'")

# n_keep = n_low = 0
# for path in paths:
#     base = os.path.splitext(os.path.basename(path))[0]
#     img = cv2.imread(path)  # BGR
#     if img is None:
#         print(f"{base}: unreadable, skipped")
#         continue
#     H, W = img.shape[:2]

#     resp = RetinaFace.detect_faces(
#         img, threshold=DET_THRESHOLD
#     )  # pass the array, detect only once
#     if not isinstance(resp, dict) or not resp:
#         print(f"{base}: no face detected")
#         continue

#     for i, (_, f) in enumerate(resp.items(), start=1):
#         x1, y1, x2, y2 = [int(v) for v in f["facial_area"]]
#         bw, bh = x2 - x1, y2 - y1
#         x1m, x2m = max(int(x1 - MARGIN * bw), 0), min(int(x2 + MARGIN * bw), W)
#         y1m, y2m = max(int(y1 - MARGIN * bh), 0), min(int(y2 + MARGIN * bh), H)

#         crop = img[y1m:y2m, x1m:x2m]  # original pixels, no resize, no rotation
#         if crop.size == 0:
#             continue

#         score = float(f["score"])
#         keep = score >= KEEP_CONF
#         folder = "raw" if keep else "low_conf"
#         name = f"{base}_face{i}"
#         cv2.imwrite(f"crops/{folder}/{name}.png", crop)

#         lms = lm_list(f["landmarks"])
#         meta = {
#             "source_image": path,
#             "detector_score": score,
#             "sent_to_shqs": keep,
#             "box_in_image": [x1m, y1m, x2m, y2m],  # crop position in the original image
#             "landmarks_in_image": lms,
#             # landmarks relative to the crop, in case SHQS wants them
#             "landmarks_in_crop": [[x - x1m, y - y1m] for x, y in lms],
#             "crop_size": [int(crop.shape[1]), int(crop.shape[0])],
#         }
#         with open(f"crops/meta/{name}.json", "w") as fh:
#             json.dump(meta, fh, indent=2)

#         if keep:
#             n_keep += 1
#         else:
#             n_low += 1
#         print(
#             f"{name}: size={crop.shape[1]}x{crop.shape[0]} score={score:.2f} -> {folder}"
#         )

# print(
#     f"\nDone: {n_keep} crops -> crops/raw (SHQS), {n_low} crops -> crops/low_conf, from {len(paths)} image(s)"
# )


def detect_and_crop(img):
    if img is None or img.size == 0:
        raise ValueError("Invalid input image")

    H, W = img.shape[:2]

    resp = RetinaFace.detect_faces(img, threshold=DET_THRESHOLD)

    results = []

    if not isinstance(resp, dict) or not resp:
        return results

    for i, (_, f) in enumerate(resp.items(), start=1):

        x1, y1, x2, y2 = [int(v) for v in f["facial_area"]]

        bw, bh = x2 - x1, y2 - y1

        x1m = max(int(x1 - MARGIN * bw), 0)
        x2m = min(int(x2 + MARGIN * bw), W)
        y1m = max(int(y1 - MARGIN * bh), 0)
        y2m = min(int(y2 + MARGIN * bh), H)

        crop = img[y1m:y2m, x1m:x2m].copy()

        if crop.size == 0:
            continue

        score = float(f["score"])
        lms = lm_list(f["landmarks"])

        results.append(
            {
                "face_id": i,
                "crop": crop,
                "detector_score": score,
                "accepted": score >= KEEP_CONF,
                "box": [x1m, y1m, x2m, y2m],
                "landmarks": [[x - x1m, y - y1m] for x, y in lms],
            }
        )

    return results
