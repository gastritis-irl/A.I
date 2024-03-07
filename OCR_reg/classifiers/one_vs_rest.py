import numpy as np


class OneVsRestLinearRegressionBinaryClassifier:
    def __init__(self, classifier_class, classes):
        self.classifier_class = classifier_class
        self.classes = classes
        self.classifiers = [classifier_class() for _ in classes]

    def fit(self, x_train, y_train):
        for idx, clf in enumerate(self.classifiers):
            y_binary = (y_train == self.classes[idx]).astype(int)
            clf.fit(x_train, y_binary)

    def predict(self, x):
        predictions = []
        for clf in self.classifiers:
            predictions.append(clf.predict(x))
        predictions = np.column_stack(predictions)
        return np.array([self.classes[i] for i in np.argmax(predictions, axis=1)])

