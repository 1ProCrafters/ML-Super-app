# algorithms/clustering/k_means.py

import random
from math import sqrt
from utils.metrics import mean_squared_error

class KMeans:
    def __init__(self, k=3, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = []

    def initialize_centroids(self, X):
        self.centroids = random.sample(X, self.k)

    def assign_clusters(self, X):
        clusters = [[] for _ in range(self.k)]
        for x in X:
            distances = [self.euclidean_distance(x, centroid) for centroid in self.centroids]
            closest_centroid = distances.index(min(distances))
            clusters[closest_centroid].append(x)
        return clusters

    def update_centroids(self, clusters):
        new_centroids = []
        for cluster in clusters:
            if cluster:
                centroid = [sum(dim) / len(cluster) for dim in zip(*cluster)]
                new_centroids.append(centroid)
            else:
                new_centroids.append(random.choice(clusters))
        self.centroids = new_centroids

    def euclidean_distance(self, a, b):
        return sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

    def fit(self, X):
        self.initialize_centroids(X)
        for iteration in range(self.max_iterations):
            clusters = self.assign_clusters(X)
            old_centroids = self.centroids.copy()
            self.update_centroids(clusters)
            if self.converged(old_centroids):
                print(f"Converged at iteration {iteration}")
                break

    def predict(self, X):
        predictions = []
        for x in X:
            distances = [self.euclidean_distance(x, centroid) for centroid in self.centroids]
            predictions.append(distances.index(min(distances)))
        return predictions

    def converged(self, old_centroids):
        for old, new in zip(old_centroids, self.centroids):
            if old != new:
                return False
        return True
