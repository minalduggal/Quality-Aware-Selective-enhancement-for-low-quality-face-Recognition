from pathlib import Path
import csv
import cv2
from embedding import generate_embedding
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

TEST_DIR = BASE_DIR / "Test_Images"
GALLERY_FILE = BASE_DIR / "reference_embeddings.npz"

# Initial experimental threshold
THRESHOLD = 0.5

# Load reference embeddings
with np.load(GALLERY_FILE, allow_pickle=False) as gallery:
    reference_embeddings = gallery["embeddings"]
    reference_labels = gallery["labels"]


def recognize_face(image):

    embedding = generate_embedding(image)

    # Cosine similarity
    similarities = reference_embeddings @ embedding

    best_index = np.argmax(similarities)
    best_score = float(similarities[best_index])

    if best_score >= THRESHOLD:
        identity = str(reference_labels[best_index])
        status = "MATCH"
    else:
        identity = "Unknown"
        status = "NO MATCH"

    return identity, best_score, status


if __name__ == "__main__":

    results = []

    for image_path in sorted(TEST_DIR.iterdir()):

        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print("Cannot read:", image_path)
            continue

        identity, score, status = recognize_face(image)

        print("\n" + "=" * 40)
        print("Test image:", image_path.name)
        print("Cosine similarity:", round(score, 4))
        print("Predicted identity:", identity)
        print("Result:", status)

        results.append([image_path.name, round(score, 4), identity, status])

    output_file = BASE_DIR / "recognition_results.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow(["Image", "Cosine_Similarity", "Predicted_Identity", "Result"])

        writer.writerows(results)

    print("\nResults saved successfully!")
    print("Saved at:", output_file)
