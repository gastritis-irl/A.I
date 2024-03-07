from nearest_centroid import NearestCentroidClassifier as NearestCentroid
from kNN import SimpleKNNClassifier as KNeighborsClassifier

def centroid_classifier(train_data, train_labels, test_data, test_labels, metric):
    clf = NearestCentroid(metric=metric)
    clf.fit(train_data, train_labels)
    train_accuracy = clf.score(train_data, train_labels)
    test_accuracy = clf.score(test_data, test_labels)
    return train_accuracy, test_accuracy


def knn_classifier(train_data, train_labels, test_data, test_labels, k, metric):
    knn = KNeighborsClassifier(k=k, metric=metric)
    knn.fit(train_data, train_labels)
    train_accuracy = knn.score(train_data, train_labels)
    test_accuracy = knn.score(test_data, test_labels)
    return train_accuracy, test_accuracy
