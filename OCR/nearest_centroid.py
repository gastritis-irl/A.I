import numpy as np
from distance_metrics import cosine_similarity_metric


def compute_centroids(x, y):
    unique_classes = np.unique(y)
    centroids = []
    for cls in unique_classes:
        mask = y == cls
        centroid = np.mean(x[mask], axis=0)
        centroids.append(centroid)
    return unique_classes, np.array(centroids)


class NearestCentroidClassifier:
    def __init__(self, metric='euclidean'):
        self.centroids = None
        self.metric = metric
        self.classes = None

    def fit(self, x, y):
        self.classes, self.centroids = compute_centroids(x, y)

    def predict(self, x):
        if self.metric == 'euclidean':
            distances = np.linalg.norm(self.centroids[:, np.newaxis] - x, axis=2)
        elif self.metric == cosine_similarity_metric:
            distances = cosine_similarity_metric(self.centroids, x)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        closest_centroid_indices = np.argmin(distances, axis=0)
        return self.classes[closest_centroid_indices]

    def score(self, x, y):
        y_pred = self.predict(x)
        return np.mean(y_pred == y)
