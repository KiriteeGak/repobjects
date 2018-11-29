import numpy as np

from scipy.spatial.distance import cdist


def proximity_based_choice(distance_matrix=None, points=None):
    """
    Distribution based choice selector. This implementation will be similar to kmeans++ like selector.
    Instead of selecting cluster centroids, we will selecting k-points out of cluster or based on some other stopping
    criterion.

    :param distance_matrix:
    :param points:
    :return:
    """
    if not distance_matrix.any() and not points:
        raise ValueError("Either distance_matrix or points are to be passed. Both cannot be None.")

    if distance_matrix.any() and isinstance(distance_matrix, list):
        distance_matrix = np.array(distance_matrix)

    if points and isinstance(points, list):
        points = np.array(points)

    if not distance_matrix.any() and points:
        distance_matrix = cdist(points, points)

    if distance_matrix.any():
        assert np.allclose(distance_matrix, distance_matrix.T, atol=1e-8), "Distance matrix should be symmetric." \
                                                                           " Found to be non-symmetric at tolerance" \
                                                                           " level 1e-8."

    array_size = len(distance_matrix)
    intra_cluster_distance = distance_matrix.sum() / (array_size * (array_size - 1))
    indices_selected = list()

    # random selection of starting index
    from random import randint
    start_index = randint(0, array_size - 1)
    indices_selected.append(start_index)

    exit_ = False

    while not exit_:
        row_ = distance_matrix[indices_selected].sum(axis=0)/len(indices_selected)

        probs = row_ / row_.sum()

        index_ = np.random.choice(array_size, 1, p=probs)[0]

        while index_ in indices_selected:
            probs[index_] = 0
            probs /= probs.sum()
            index_ = np.random.choice(array_size, 1, p=probs)[0]

            if row_[index_].max() < intra_cluster_distance:
                exit_ = True

        indices_selected.append(index_)

    return indices_selected


def maximum_proximity(distance_matrix):
    array_size = len(distance_matrix)
    intra_cluster_distance = distance_matrix.sum()/(array_size * (array_size-1))

    indices_selected = list()

    # random selection of starting index
    from random import randint
    start_index = randint(0, array_size - 1)
    indices_selected.append(start_index)

    exit_ = False

    while not exit_:
        row_ = distance_matrix[indices_selected].sum(axis=0)/len(indices_selected)

        index_ = np.where(row_ == row_.max())[0][0]

        while index_ in indices_selected:
            row_[index_] = 0
            index_ = np.where(row_ == row_.max())[0][0]
            if row_.max() < intra_cluster_distance:
                exit_ = True

        if not exit_:
            indices_selected.append(index_)

    return indices_selected


# arr = np.array([[3, 3], [-3, 3], [-3, -3], [3, -3], [1, 1], [-1, 1], [1, -1], [-1, -1]])
# sample_distance_matrix = cdist(arr, arr)
# print(proximity_based_choice(sample_distance_matrix))
