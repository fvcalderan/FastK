import numpy as np
import pandas as pd
from typing import Union
from functools import reduce

class FastK:
    """FastK K-Medoids algorithm implementation in Python"""

    def __init__(self, n_clusters : int = 8):
        """Constructor method for FastK

        parameters
        ----------
        n_clusters : int
            number of seeds to be placed and, consequently, the number of
            clusters to be formed
        """
        self.n_clusters_ = n_clusters
        self.labels_  = None
        self.medoids_ = None
        self.dist_    = None
        self.v_       = None

    def fit(self, df : Union[pd.DataFrame, np.ndarray]):
        """Fit the dataset to get the labels, medoids and distances

        parameters
        ----------
        df : Union[pd.DataFrame, np.ndarray]
            dataset to cluster. Although it's recommended that it's either a
            numpy.ndarray or pandas.DataFrame, anything castable to
            numpy.array should work correctly if it has the correct dimensions
            (2x2) and data types (numerical values)
        """
        if type(df) == np.ndarray: D = df
        else:                      D = np.array(df)
        n_elems = D.shape[0]

        # Step 1. Fill the distance matrix
        self.dist_ = np.zeros((n_elems, n_elems))
        for i in range(n_elems):
            for j in range(n_elems):
                self.dist_[i][j] = np.linalg.norm(D[i]-D[j])

        # Step 2. Calculate vector v
        self.v_ = np.zeros(n_elems)
        for j in range(n_elems):
            j_sum = 0
            for i in range(n_elems):
                l_sum = reduce(lambda xi, xl : xi+xl, self.dist_[i])
                j_sum += self.dist_[i][j]/l_sum
            self.v_[j] = j_sum

        # Step 3. Medoids = indices for the n_clusters smallest values in v
        self.medoids_ = np.argsort(self.v_)[:self.n_clusters_]

        # Step 4. Assign each object to the nearest medoid
        self.labels_ = np.zeros(n_elems)
        arr_m_dist   = np.zeros(n_elems)
        for i in range(n_elems):
            d_to_medoids = np.array([self.dist_[i][j] for j in self.medoids_])
            self.labels_[i] = np.argmin(d_to_medoids)
            arr_m_dist[i] = np.min(d_to_medoids)

        # Step 5. Calculate the sum of d(medoids, elements)
        sum_of_dists = np.sum(arr_m_dist)
        new_sum_of_dists = -1

        while sum_of_dists != new_sum_of_dists:
            sum_of_dists = new_sum_of_dists
            # Step 6. Find new medoid for each cluster, minimizing total
            # distance to other objects in its cluster. Update medoids
            for clus in range(len(self.medoids_)):
                # get elements that belong in this cluster
                elems_i = np.where(self.labels_ == clus)[0]

                # this will store the best medoid so far for this cluster
                partial_tuple = (0, 99999999)   # index, sum of distances

                # get new best medoid
                for temp_m in elems_i:
                    # calculate d(temp_medoid, elems in clus)
                    dist_s = sum([self.dist_[temp_m][i] for i in elems_i])
                    if partial_tuple[1] > dist_s:
                        partial_tuple = (temp_m, dist_s)

                self.medoids_[clus] = partial_tuple[0]

            # Step 7. Assign each object to the nearest medoid
            for i in range(n_elems):
                d_to_medoids = np.array([self.dist_[i][j]
                                         for j in self.medoids_])
                self.labels_[i] = np.argmin(d_to_medoids)
                arr_m_dist[i] = np.min(d_to_medoids)

            # calculate sum of d(elements, medoids)
            new_sum_of_dists = np.sum(arr_m_dist)
