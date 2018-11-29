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
    if not distance_matrix and not points:
        raise ValueError("Either distance_matrix or points are to be passed. Both cannot be None.")

    if distance_matrix and isinstance(distance_matrix, list):
        distance_matrix = np.array(distance_matrix)

    if points and isinstance(points, list):
        points = np.array(points)

    if not distance_matrix and points:
        distance_matrix = cdist(points, points)

    if distance_matrix:
        assert np.allclose(distance_matrix, distance_matrix.T, atol=1e-8), "Distance matrix should be symmetric." \
                                                                           " Found to be non-symmetric at tolerance" \
                                                                           " level 1e-8."

    array_size = len(distance_matrix)
    indices_selected = list()

    # random selection of starting index
    from random import randint
    start_index = randint(0, array_size - 1)
    indices_selected.append(start_index)

    exit_ = False

    while not exit_:
        row_ = distance_matrix[indices_selected]

        probs = row_.sum(axis=0) / row_.sum()

        if probs.max() - probs.min() <= 10e-3:
            exit_ = True

        index_ = np.random.choice(array_size, 1, p=probs)[0]

        while index_ in indices_selected:
            probs[index_] = 0
            probs /= probs.sum()
            index_ = np.random.choice(array_size, 1, p=probs)[0]

        indices_selected.append(index_)

    return indices_selected


def maximum_proximity():
    pass
