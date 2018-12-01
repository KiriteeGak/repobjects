import numpy as np

from scipy.spatial.distance import cdist


class GetObjects(object):
    """
    A method to select objects from a cluster to represent the cluster.
    """
    def __init__(self, method="choice", regularize=None, start_index="random"):
        """
        :param method:
            Options:
                * choice: Distribution based choice selector. This implementation will be similar to kmeans++
                          like selector. Instead of selecting cluster centroids, we will selecting k-points out
                          of cluster or based on some other stopping criterion.
                  max_distance: Selects points based on maximum avg. distance from the currently selected objects.
        :param regularize: Better usage when method is choice.
                           Helps regularising in selecting less points from dense clusters.
                           Recommended value between 1 and 3.
        :param start_index: Optional. Start index object to be selected.
                            If not random choice is picked from the cluster.
        """

        self.method = method
        self.regularize = regularize
        self.start_index = start_index

    def extract(self, distance_matrix=None, point_matrix=None):
        if not distance_matrix.any() and not point_matrix:
            raise ValueError("Either distance_matrix or points are to be passed. Both cannot be None.")

        if point_matrix and isinstance(point_matrix, list):
            point_matrix = np.array(point_matrix)

        if point_matrix and not isinstance(distance_matrix, np.ndarray) and not distance_matrix:
            distance_matrix = cdist(point_matrix, point_matrix)

        if distance_matrix.any():
            assert np.allclose(distance_matrix,
                               distance_matrix.T,
                               atol=1e-8), "Distance matrix should be symmetric." \
                                           " Found to be non-symmetric at tolerance level 1e-8."

        array_size = len(distance_matrix)
        intra_cluster_distance = distance_matrix.sum() / (array_size * (array_size - 1))
        indices_selected = list()

        if self.start_index == "random":

            # random selection of starting index
            from random import randint
            start_index = randint(0, array_size - 1)
        else:
            assert isinstance(self.start_index, int), "Start index should be of type `int`"
            assert self.start_index < array_size, "Start index is greater than the array size passed."

            start_index = self.start_index

        indices_selected.append(start_index)

        exit_ = False

        if self.method == 'choice':
            while not exit_:
                row_ = distance_matrix[indices_selected].sum(axis=0) / len(indices_selected)
                probs = row_ / row_.sum()
                index_ = np.random.choice(array_size, 1, p=probs)[0]

                while index_ in indices_selected:
                    probs[index_] = 0
                    probs /= probs.sum()
                    index_ = np.random.choice(array_size, 1, p=probs)[0]

                    if row_[index_].max() < intra_cluster_distance:
                        exit_ = True
                if not exit_:
                    indices_selected.append(index_)

        elif self.method == 'max_distance':
            while not exit_:
                row_ = distance_matrix[indices_selected].sum(axis=0) / len(indices_selected)
                index_ = np.where(row_ == row_.max())[0][0]

                while index_ in indices_selected:
                    row_[index_] = 0
                    index_ = np.where(row_ == row_.max())[0][0]
                    if row_.max() < intra_cluster_distance:
                        exit_ = True

                if not exit_:
                    indices_selected.append(index_)
        else:
            raise ValueError("Unable to find the method {method} passed to select. "
                             "Possible options are `choice` and `max_distance`")

        return indices_selected


# arr = np.array([[3, 3], [-3, 3], [-3, -3], [3, -3], [1, 1], [-1, 1], [1, -1], [-1, -1]])
# sample_distance_matrix = cdist(arr, arr)
# go = GetObjects()
# print(go.extract(sample_distance_matrix))
