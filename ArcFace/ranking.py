from pathlib import Path
import numpy as np

if __package__:
    from .embedding import generate_embedding
else:
    from embedding import generate_embedding


BASE_DIR = Path(__file__).resolve().parent
GALLERY_FILE = BASE_DIR / "reference_embeddings.npz"

THRESHOLD = 0.5

reference_embeddings = None
reference_labels = None


def load_gallery():

    global reference_embeddings, reference_labels

    if reference_embeddings is None:

        with np.load(GALLERY_FILE, allow_pickle=False) as gallery:

            reference_embeddings = gallery["embeddings"]
            reference_labels = gallery["labels"]

    return reference_embeddings, reference_labels


def rank_faces(image, top_k=5, threshold=THRESHOLD):

    embeddings, labels = load_gallery()

    query_embedding = generate_embedding(image)

    # Cosine similarity
    similarities = np.clip(embeddings @ query_embedding, -1, 1)

    # Best score for each unique identity
    identity_scores = {}

    for label, score in zip(labels, similarities):

        label = str(label)
        score = float(score)

        if label not in identity_scores:
            identity_scores[label] = score

        else:
            identity_scores[label] = max(identity_scores[label], score)

    # Sort identities by similarity
    ranked = sorted(identity_scores.items(), key=lambda item: item[1], reverse=True)

    if not ranked:
        raise ValueError("Reference gallery is empty")

    top_matches = []

    for rank, (identity, score) in enumerate(ranked[:top_k], start=1):

        top_matches.append(
            {"rank": rank, "identity": identity, "similarity": round(score, 4)}
        )

    best_identity, best_score = ranked[0]

    matched = best_score >= threshold

    return {
        "predicted_identity": best_identity if matched else "Unknown",
        "similarity": round(best_score, 4),
        "status": "MATCH" if matched else "NO MATCH",
        "threshold": threshold,
        "top_matches": top_matches,
    }
