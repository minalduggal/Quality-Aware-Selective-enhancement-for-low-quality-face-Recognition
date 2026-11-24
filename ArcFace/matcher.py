import numpy as np
from pathlib import Path
from arcface import generate_embedding

GALLERY = Path(r"C:\Users\Dell\OneDrive\Desktop\tinyface\Testing_Set\Gallery_Match")


def compare(image1, image2):
    emb1 = generate_embedding(str(GALLERY / image1))
    emb2 = generate_embedding(str(GALLERY / image2))

    if emb1 is None or emb2 is None:
        print("Embedding extraction failed")
        return

    similarity = float(np.dot(emb1, emb2))

    print(f"{image1} vs {image2}: {similarity:.4f}")


print("Same person:")
compare("1_106.jpg", "1_118.jpg")

print("\nDifferent people:")
compare("1_106.jpg", "2_9.jpg")
