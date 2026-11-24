from pathlib import Path
import numpy as np
from scipy.io import loadmat
from arcface import generate_embedding

ROOT = Path(r"C:\Users\Dell\OneDrive\Desktop\tinyface\Testing_Set")

gallery_data = loadmat(ROOT / "gallery_match_img_ID_pairs.mat")
probe_data = loadmat(ROOT / "probe_img_ID_pairs.mat")


def read_names(data, key):
    return [str(x.item()) for x in data[key].flatten()]


gallery_names = read_names(gallery_data, "gallery_set")
probe_names = read_names(probe_data, "probe_set")

gallery_ids = gallery_data["gallery_ids"].flatten()
probe_ids = probe_data["probe_ids"].flatten()

# Select 10 identities and their corresponding images
selected_ids = list(dict.fromkeys(probe_ids))[:10]

gallery_indices = [
    i for i, identity in enumerate(gallery_ids) if identity in selected_ids
]

probe_indices = [i for i, identity in enumerate(probe_ids) if identity in selected_ids]


def extract(folder, names, ids, indices):
    features, labels = [], []

    for i in indices:
        path = ROOT / folder / names[i]
        embedding = generate_embedding(str(path))

        if embedding is not None:
            features.append(embedding)
            labels.append(ids[i])

    return np.asarray(features), np.asarray(labels)


print("Extracting gallery embeddings...")
gallery_features, gallery_labels = extract(
    "Gallery_Match", gallery_names, gallery_ids, gallery_indices
)

print("Extracting probe embeddings...")
probe_features, probe_labels = extract("Probe", probe_names, probe_ids, probe_indices)

# Embeddings are already L2-normalized
similarities = probe_features @ gallery_features.T

rank1_correct = 0
rank5_correct = 0

for i, scores in enumerate(similarities):
    ranked_indices = np.argsort(scores)[::-1]

    top1_ids = gallery_labels[ranked_indices[:1]]
    top5_ids = gallery_labels[ranked_indices[:5]]

    if probe_labels[i] in top1_ids:
        rank1_correct += 1

    if probe_labels[i] in top5_ids:
        rank5_correct += 1

total = len(probe_labels)

print("\n===== TINYFACE SMALL TEST =====")
print("Identities:", len(selected_ids))
print("Gallery images:", len(gallery_labels))
print("Probe images:", total)
print(f"Rank-1 Accuracy: {rank1_correct / total * 100:.2f}%")
print(f"Rank-5 Accuracy: {rank5_correct / total * 100:.2f}%")
