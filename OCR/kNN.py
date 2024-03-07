import numpy as np
from collections import Counter
from distance_metrics import euclidean_distance

class SimpleKNNClassifier:
    def __init__(self, k=3, metric='euclidean'):
        self.k = k
        self.metric = metric
        self.train_data = None
        self.train_labels = None

    def fit(self, train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels

    def predict(self, test_data, metric_func=euclidean_distance):
        num_test_samples = test_data.shape[0]
        num_train_samples = self.train_data.shape[0]
        distances = np.zeros((num_test_samples, num_train_samples))

        for i, test_sample in enumerate(test_data):
            for j, train_sample in enumerate(self.train_data):
                distances[i, j] = metric_func(test_sample, train_sample)

        knn_indices = np.argpartition(distances, self.k)[:, :self.k]
        knn_labels = self.train_labels[knn_indices]
        return np.array([Counter(row_labels).most_common(1)[0][0] for row_labels in knn_labels])

    def score(self, test_data, test_labels):
        predictions = self.predict(test_data)
        return np.mean(predictions == test_labels)
