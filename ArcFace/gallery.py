from pathlib import Path
from scipy.io import loadmat
from arcface import generate_embedding

DATASET_ROOT = Path(r"C:\Users\Dell\OneDrive\Desktop\tinyface")

GALLERY_FOLDER = DATASET_ROOT / "Testing_Set" / "Gallery_Match"
MAT_FILE = DATASET_ROOT / "Testing_Set" / "gallery_match_img_ID_pairs.mat"

# Load official TinyFace image ordering
data = loadmat(MAT_FILE)
gallery_names = [str(item.item()) for item in data["gallery_set"].flatten()]

# Test only 5 images
for i, name in enumerate(gallery_names[:5], start=1):
    image_path = GALLERY_FOLDER / name

    print(f"\nProcessing image {i}: {name}")

    embedding = generate_embedding(str(image_path))

    if embedding is not None:
        print("Embedding shape:", embedding.shape)
    else:
        print("Face detection failed")
