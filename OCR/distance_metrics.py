import numpy as np

def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

def cosine_similarity(v1, v2):
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    return np.dot(v1, v2) / (norm_v1 * norm_v2)

def cosine_similarity_metric(X, Y):
    if X.ndim == 1:
        X = X[np.newaxis, :]
    if Y.ndim == 1:
        Y = Y[np.newaxis, :]

    X_norm = np.linalg.norm(X, axis=1)[:, np.newaxis]
    Y_norm = np.linalg.norm(Y, axis=1)[np.newaxis, :]
    dot_product = np.dot(X, Y.T)
    cosine_similarities = dot_product / (X_norm * Y_norm)

    # Convert cosine similarities to a distance-like metric
    cosine_distances = 1 - cosine_similarities

    return cosine_distances



