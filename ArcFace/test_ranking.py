from pathlib import Path
import cv2
import json
from ranking import rank_faces

BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "Test_Images"

for image_path in sorted(TEST_DIR.iterdir()):

    if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        continue

    result = rank_faces(image, top_k=5)

    print("\n" + "=" * 50)
    print("Test image:", image_path.name)
    print(json.dumps(result, indent=4))
