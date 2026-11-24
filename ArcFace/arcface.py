import cv2
from insightface.app import FaceAnalysis

# Load pretrained ArcFace model
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])

app.prepare(ctx_id=-1, det_size=(640, 640))


import cv2
import numpy as np
from insightface.utils import face_align


def generate_embedding(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not load image:", image_path)
        return None

    # TinyFace images are already cropped faces
    # Resize to ArcFace's expected input dimensions
    face_crop = cv2.resize(image, (112, 112), interpolation=cv2.INTER_CUBIC)

    # Use recognition model directly (skip face detection)
    recognition_model = app.models["recognition"]

    embedding = recognition_model.get_feat(face_crop)

    embedding = np.asarray(embedding).reshape(-1)

    # Normalize embedding
    norm = np.linalg.norm(embedding)

    if norm == 0:
        return None

    embedding = embedding / norm

    return embedding.astype(np.float32)
