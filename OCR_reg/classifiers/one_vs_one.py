import numpy as np
from itertools import combinations


class OneVsOneLinearRegressionBinaryClassifier:
    def __init__(self, classifier_class, classes):
        self.classifier_class = classifier_class
        self.classes = classes
        self.classifiers = []
        self.class_combinations = list(combinations(classes, 2))
        for _ in self.class_combinations:
            self.classifiers.append(classifier_class())

    def fit(self, x_train, y_train):
        for idx, clf in enumerate(self.classifiers):
            class_a, class_b = self.class_combinations[idx]
            mask = np.isin(y_train, [class_a, class_b])
            x_train_subset = x_train[mask]
            y_train_subset = y_train[mask]
            y_binary = (y_train_subset == class_b).astype(int)
            clf.fit(x_train_subset, y_binary)

    def predict(self, x_test):
        predictions_matrix = np.column_stack([clf.predict(x_test) for clf in self.classifiers])
        predictions = []
        for pred in predictions_matrix:
            class_votes = {cls: 0 for cls in self.classes}
            for idx, p in enumerate(pred):
                winner = self.class_combinations[idx][int(p)]
                class_votes[winner] += 1
            predictions.append(max(class_votes, key=class_votes.get))
        return np.array(predictions)
