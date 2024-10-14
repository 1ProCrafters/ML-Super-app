# algorithms/clustering/dbscan.py

from math import sqrt

class Dbscan:
    def __init__(self, eps=0.5, min_pts=5):
        self.eps = eps
        self.min_pts = min_pts
        self.labels = []

    def fit(self, X):
        self.labels = [0] * len(X)
        cluster_id = 0

        for idx in range(len(X)):
            if self.labels[idx] != 0:
                continue
            neighbors = self._region_query(X, idx)
            if len(neighbors) < self.min_pts:
                self.labels[idx] = -1  # Noise
            else:
                cluster_id += 1
                self._expand_cluster(X, idx, neighbors, cluster_id)

    def predict(self, X):
        return self.labels

    def _expand_cluster(self, X, idx, neighbors, cluster_id):
        self.labels[idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if self.labels[neighbor_idx] == -1:
                self.labels[neighbor_idx] = cluster_id
            elif self.labels[neighbor_idx] == 0:
                self.labels[neighbor_idx] = cluster_id
                new_neighbors = self._region_query(X, neighbor_idx)
                if len(new_neighbors) >= self.min_pts:
                    neighbors += new_neighbors
            i += 1

    def _region_query(self, X, idx):
        neighbors = []
        for i, point in enumerate(X):
            if self._euclidean_distance(X[idx], point) <= self.eps:
                neighbors.append(i)
        return neighbors

    def _euclidean_distance(self, a, b):
        return sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
