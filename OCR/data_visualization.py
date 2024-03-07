import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from distance_metrics import cosine_similarity


def plot_digit(data):
    image = data.reshape(8, 8)
    plt.imshow(image, cmap='gray')
    plt.show()


def cosine_similarity_matrix(data):
    similarity_matrix = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        for j in range(len(data)):
            similarity_matrix[i, j] = cosine_similarity(data[i], data[j])
    return similarity_matrix


def plot_heatmap(test_labels, cosine_similarity_matrix):
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Group data points by labels
    label_groups = {label: [] for label in range(10)}
    for i, label in enumerate(test_labels):
        label_groups[label].append(i)

    # Calculate cosine similarities within each group
    grouped_cosine_similarity_matrix = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            indices_i = label_groups[i]
            indices_j = label_groups[j]
            avg_cosine_similarity = np.mean(cosine_similarity_matrix[np.ix_(indices_i, indices_j)])
            grouped_cosine_similarity_matrix[i, j] = avg_cosine_similarity

    # Create the heatmap
    sns.heatmap(grouped_cosine_similarity_matrix, annot=True, cmap='plasma', vmin=-1, vmax=1)
    plt.xlabel("Labels")
    plt.ylabel("Labels")
    plt.title("Cosine Similarities Grouped by Labels")
    plt.show()
