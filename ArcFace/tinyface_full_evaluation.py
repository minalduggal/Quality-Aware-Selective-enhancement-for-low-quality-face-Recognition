from pathlib import Path
import argparse
import csv
import json

import cv2
import numpy as np
from scipy.io import loadmat, savemat
from insightface.app import FaceAnalysis

ROOT = Path(r"C:\Users\Dell\OneDrive\Desktop\tinyface")
TEST = ROOT / "Testing_Set"
EVAL = ROOT / "Face_Identification_Evaluation"
OUTPUT = EVAL / "arcface_baseline"

DIM = 512
BATCH_SIZE = 128

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))
recognizer = app.models["recognition"]


def read_mapping(filename, name_key, id_key):
    data = loadmat(TEST / filename)

    names = [str(x.item()) for x in data[name_key].flatten()]

    ids = data[id_key].flatten().astype(int)

    return names, ids


def get_datasets():
    gallery_names, gallery_ids = read_mapping(
        "gallery_match_img_ID_pairs.mat", "gallery_set", "gallery_ids"
    )

    probe_names, probe_ids = read_mapping(
        "probe_img_ID_pairs.mat", "probe_set", "probe_ids"
    )

    distractor_names = sorted(
        p.name
        for p in (TEST / "Gallery_Distractor").iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    return {
        "gallery": (TEST / "Gallery_Match", gallery_names, gallery_ids),
        "probe": (TEST / "Probe", probe_names, probe_ids),
        "distractor": (TEST / "Gallery_Distractor", distractor_names, None),
    }


def embed(path):
    image = cv2.imread(str(path))

    if image is None:
        return None

    # TinyFace contains cropped faces.
    # Direct resize is our current baseline preprocessing.
    crop = cv2.resize(image, (112, 112), interpolation=cv2.INTER_CUBIC)

    feature = recognizer.get_feat(crop)
    feature = np.asarray(feature, dtype=np.float32).reshape(-1)

    if feature.size != DIM:
        return None

    norm = np.linalg.norm(feature)

    if not np.isfinite(norm) or norm < 1e-12:
        return None

    return feature / norm


def extract_group(key, folder, names):
    OUTPUT.mkdir(parents=True, exist_ok=True)

    feature_path = OUTPUT / f"{key}.npy"
    progress_path = OUTPUT / f"{key}_progress.json"
    failure_path = OUTPUT / f"{key}_failures.txt"

    total = len(names)

    if feature_path.exists():
        features = np.lib.format.open_memmap(feature_path, mode="r+")
        if features.shape != (total, DIM):
            raise ValueError(f"Existing {key} feature shape mismatch")
    else:
        features = np.lib.format.open_memmap(
            feature_path, mode="w+", dtype=np.float32, shape=(total, DIM)
        )
        features[:] = np.nan
        features.flush()

    start = 0

    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        start = int(progress["completed"])

    print(f"\nProcessing {key}: {start}/{total}")

    with open(failure_path, "a", encoding="utf-8") as log:
        for i in range(start, total):
            path = folder / names[i]

            try:
                feature = embed(path)
            except Exception as exc:
                print(f"Error: {path.name}: {exc}")
                feature = None

            if feature is None:
                features[i] = np.nan
                log.write(f"{i},{names[i]}\n")
            else:
                features[i] = feature

            if (i + 1) % 100 == 0 or i + 1 == total:
                features.flush()

                progress_path.write_text(json.dumps({"completed": i + 1}))

                print(f"{key}: {i + 1}/{total}")

    features.flush()
    print(f"{key} extraction complete")


def extract_all(datasets):
    for key in ("gallery", "probe", "distractor"):
        folder, names, _ = datasets[key]
        extract_group(key, folder, names)


def load_features(key):
    path = OUTPUT / f"{key}.npy"

    if not path.exists():
        raise FileNotFoundError(path)

    features = np.load(path, mmap_mode="r")

    progress_path = OUTPUT / f"{key}_progress.json"
    if not progress_path.exists():
        raise RuntimeError(f"{key} extraction incomplete")

    completed = json.loads(progress_path.read_text())["completed"]

    if completed != len(features):
        raise RuntimeError(f"{key} extraction incomplete")

    return features


def export_mat_files():
    features_folder = EVAL / "features"
    features_folder.mkdir(parents=True, exist_ok=True)

    mapping = {
        "gallery": "gallery_feature_map",
        "probe": "probe_feature_map",
        "distractor": "distractor_feature_map",
    }

    for key, variable in mapping.items():
        features = load_features(key)

        savemat(
            features_folder / f"{key}.mat",
            {variable: np.asarray(features)},
            do_compression=False,
        )

        print(f"Saved {key}.mat")


def average_precision(relevant):
    relevant = np.asarray(relevant, dtype=bool)

    ngood = int(relevant.sum())

    if ngood == 0:
        return 0.0

    old_recall = 0.0
    old_precision = 1.0
    intersect_size = 0
    ap = 0.0

    for rank, is_good in enumerate(relevant, start=1):
        if is_good:
            intersect_size += 1

        recall = intersect_size / ngood
        precision = intersect_size / rank

        ap += (recall - old_recall) * ((old_precision + precision) / 2)

        old_recall = recall
        old_precision = precision

        if intersect_size == ngood:
            break

    return float(ap)


def evaluate(datasets):
    OUTPUT.mkdir(parents=True, exist_ok=True)

    gallery = load_features("gallery")
    probe = load_features("probe")
    distractor = load_features("distractor")

    gallery_ids = datasets["gallery"][2]
    probe_ids = datasets["probe"][2]

    if len(gallery) != len(gallery_ids):
        raise ValueError("Gallery mapping mismatch")

    if len(probe) != len(probe_ids):
        raise ValueError("Probe mapping mismatch")

    all_gallery = np.concatenate((gallery, distractor), axis=0)

    valid_gallery = np.isfinite(all_gallery).all(axis=1)

    valid_probe = np.isfinite(probe).all(axis=1)

    # Distractors have no matching probe identity.
    all_ids = np.concatenate((gallery_ids, np.full(len(distractor), -1, dtype=int)))

    print("Gallery:", len(gallery))
    print("Distractors:", len(distractor))
    print("Probes:", len(probe))

    print("Invalid gallery embeddings:", int((~valid_gallery).sum()))
    print("Invalid probe embeddings:", int((~valid_probe).sum()))

    if not valid_gallery.all() or not valid_probe.all():
        raise RuntimeError(
            "Some embeddings failed. Do not report "
            "official benchmark accuracy until all "
            "images have valid features."
        )

    correct = {1: 0, 5: 0, 10: 0, 20: 0, 50: 0}
    ap_scores = []

    results = []

    total = len(probe)

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)

        batch = np.asarray(probe[start:end], dtype=np.float32)

        # Cosine similarity because embeddings
        # have already been L2-normalized.
        scores = batch @ all_gallery.T

        for local_index, row in enumerate(scores):
            index = start + local_index
            true_id = probe_ids[index]

            ranked = np.argsort(row)[::-1]
            ranked_ids = all_ids[ranked]

            relevant = ranked_ids == true_id

            for k in correct:
                if np.any(relevant[:k]):
                    correct[k] += 1

            ap = average_precision(relevant)
            ap_scores.append(ap)

            results.append(
                {
                    "probe_image": datasets["probe"][1][index],
                    "true_id": int(true_id),
                    "top1_id": int(ranked_ids[0]),
                    "top1_similarity": float(row[ranked[0]]),
                    "correct_rank1": int(relevant[0]),
                    "AP": ap,
                }
            )

        print(f"Evaluated {end}/{total} probes")

    metrics = {f"Rank-{k}": correct[k] / total * 100 for k in correct}

    metrics["mAP"] = float(np.mean(ap_scores) * 100)

    print("\n===== TINYFACE RESULTS =====")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}%")

    with open(OUTPUT / "results.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=4))

    print("\nResults saved in:", OUTPUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("action", choices=["extract", "evaluate", "export"])

    args = parser.parse_args()

    datasets = get_datasets()

    if args.action == "extract":
        extract_all(datasets)

    elif args.action == "evaluate":
        evaluate(datasets)

    elif args.action == "export":
        export_mat_files()
