from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

BASE_DIR = Path(__file__).resolve().parent

GALLERY_DIR = BASE_DIR / "Reference_Gallery"
OUTPUT_FILE = BASE_DIR / "reference_embeddings.npz"

# Load recognition model only
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])

app.prepare(ctx_id=-1, det_size=(640, 640))

recognizer = app.models["recognition"]

embeddings = []
labels = []


def generate_embedding(image):

    if image is None or image.size == 0:
        raise ValueError("Invalid image")

    # Resize already cropped face
    face = cv2.resize(image, (112, 112), interpolation=cv2.INTER_CUBIC)

    # Generate ArcFace embedding
    embedding = recognizer.get_feat(face).flatten()
    embedding = embedding.astype(np.float32)

    # L2 normalization
    norm = np.linalg.norm(embedding)

    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Invalid embedding")

    return embedding / norm


for person_folder in sorted(GALLERY_DIR.iterdir()):

    if not person_folder.is_dir():
        continue

    for image_path in sorted(person_folder.iterdir()):

        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print("Cannot read:", image_path)
            continue

        embedding = generate_embedding(image)

        embeddings.append(embedding)
        labels.append(person_folder.name)

        print("\nImage:", image_path.name)
        print("Identity:", person_folder.name)
        print("Original dimensions:", image.shape)
        print("Embedding shape:", embedding.shape)
        print("Embedding generated successfully!")


if embeddings:

    np.savez_compressed(
        OUTPUT_FILE, embeddings=np.array(embeddings), labels=np.array(labels)
    )

    print("\nReference gallery created successfully!")
    print("Total reference images:", len(embeddings))
    print("Total identities:", len(set(labels)))
    print("Saved at:", OUTPUT_FILE)

else:
    print("No valid reference embeddings generated.")
