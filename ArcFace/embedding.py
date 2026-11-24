import cv2
import numpy as np
from insightface.app import FaceAnalysis

recognizer = None


def load_model():

    global recognizer

    if recognizer is None:

        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])

        app.prepare(ctx_id=-1, det_size=(640, 640))

        recognizer = app.models["recognition"]

    return recognizer


def generate_embedding(image):

    if image is None or image.size == 0:
        raise ValueError("Invalid input image")

    model = load_model()

    # Resize cropped face
    face = cv2.resize(image, (112, 112), interpolation=cv2.INTER_CUBIC)

    # Generate 512-dimensional embedding
    embedding = model.get_feat(face).flatten()
    embedding = embedding.astype(np.float32)

    # L2 normalization
    norm = np.linalg.norm(embedding)

    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Invalid embedding")

    return embedding / norm


def cosine_similarity(embedding1, embedding2):

    return float(np.clip(np.dot(embedding1, embedding2), -1, 1))
