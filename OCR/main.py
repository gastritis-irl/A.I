import pandas as pd
from nearest_centroid import NearestCentroidClassifier as NearestCentroid
from kNN import SimpleKNNClassifier as KNeighborsClassifier
from distance_metrics import euclidean_distance, cosine_similarity, cosine_similarity_metric
from classifiers import knn_classifier, centroid_classifier
from data_visualization import plot_digit, cosine_similarity_matrix, plot_heatmap


def load_data(train_file, test_file):
    train_data = pd.read_csv(train_file, header=None)
    test_data = pd.read_csv(test_file, header=None)

    train_labels = train_data.iloc[:, -1].values
    train_data = train_data.iloc[:, :-1].values

    test_labels = test_data.iloc[:, -1].values
    test_data = test_data.iloc[:, :-1].values

    return train_data, train_labels, test_data, test_labels


def main():
    train_file = 'optdigits.tra'
    test_file = 'optdigits.tes'

    train_data, train_labels, test_data, test_labels = load_data(train_file, test_file)

    k = 3

    for metric in ['euclidean', cosine_similarity_metric]:
        train_accuracy, test_accuracy = knn_classifier(train_data, train_labels, test_data, test_labels, k, metric)
        print(f'kNN ({metric}): Train accuracy: {train_accuracy}, Test accuracy: {test_accuracy}')

        train_accuracy, test_accuracy = centroid_classifier(train_data, train_labels, test_data, test_labels, metric)
        print(f'Centroid ({metric}): Train accuracy: {train_accuracy}, Test accuracy: {test_accuracy}')

    clf = NearestCentroid()
    clf.fit(train_data, train_labels)
    for i in range(10):
        plot_digit(clf.centroids[i])

    similarity_matrix = cosine_similarity_matrix(test_data)
    plot_heatmap(test_labels, similarity_matrix)


if __name__ == '__main__':
    main()
