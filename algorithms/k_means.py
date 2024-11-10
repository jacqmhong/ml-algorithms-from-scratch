# k-Means Clustering

import numpy as np
from utils.distance_metrics import euclidean_distance

class KMeans:
    def __init__(self, num_clusters=3, max_iter=100, change_tol=1e-4):
        """
        Initializes the k-means clustering classifier.

        Parameters:
            num_clusters (int): The number of clusters to partition the data into.
            max_iter (int): The maximum number of iterations allowed for the k-means algorithm to converge.
            change_tol (float): The tolerance level for changes in centroid positions, used to determine convergence.
        """
        self.num_clusters = num_clusters  # k
        self.max_iter = max_iter
        self.change_tol = change_tol
        self.centroids = None  # ndarray after fitting

    def fit(self, X, y=None):
        """
        Performs k-means clustering.

        Parameters:
            X (ndarray): Feature matrix of shape (n_samples, n_features).
            y (ndarray): Target vector of shape (n_samples,).
                * y is not used - it is here to maintain consistency among classifiers
        """
        # Initialize the starting centroids
        self._init_centroids_kmeans_plus(X)

        for _ in range(self.max_iter):
            # Assign each sample of X to its nearest centroid (cluster)
            self.labels = self._assign_clusters(X)

            # Recalculate the centroids based on the assignments
            new_centroids = self._calculate_centroids(X)

            # Break if converged (if all centroid changes are less than the tolerance threshold)
            if np.all(np.abs(new_centroids - self.centroids) < self.change_tol):
                break
            self.centroids = new_centroids

    def _init_centroids_random(self, X):
        """
        Randomly initializes centroids by picking getting indices of X and selecting the corresponding samples.

        Parameters:
            X (ndarray): Feature matrix of shape (n_samples, n_features).
        """
        random_indices = np.random.choice(len(X), self.num_clusters, replace=False)
        self.centroids = X[random_indices]

    def _init_centroids_kmeans_plus(self, X):
        """
        An optimized initialization of centroids by using the kmeans++ algorithm.

        Parameters:
            X (ndarray): Feature matrix of shape (n_samples, n_features).
        """
        # The first centroid is initialized randomly
        self.centroids = [X[np.random.choice(len(X))]]

        for _ in range(1, self.num_clusters):
            # Distances from each point to the nearest centroid
            distances = np.empty(len(X))
            for i, x in enumerate(X):
                distances[i] = min(np.linalg.norm(x - c) ** 2 for c in self.centroids)

            probabilities = distances / distances.sum() # p = distance_i / sum of distances
            cumulative_probs = np.cumsum(probabilities)

            # Ensures that centroids are spread out via simulated weighted random sampling
            r = np.random.rand()
            next_centroid = X[np.searchsorted(cumulative_probs, r)]
            self.centroids.append(next_centroid)

        self.centroids = np.array(self.centroids)

    def _assign_clusters(self, X):
        """
        Assigns each sample in X to its nearest cluster based on the current centroids.

        Parameters:
            X (ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            labels (ndarray): An array of shape (n_samples,) containing the index of the nearest cluster for each sample in X.
        """
        # Calculate distances from each sample to each centroid
        distances = []
        for sample in X:
            sample_distances = [euclidean_distance(sample, centroid) for centroid in self.centroids]
            distances.append(sample_distances)

        distances = np.array(distances)
        return np.argmin(distances, axis=1)

    def _calculate_centroids(self, X):
        """
        Calculates the new centroids based on the current cluster assignments.

        Parameters:
            X (ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            new_centroids (ndarray): Array of shape (num_clusters, n_features) containing the updated centroids for each cluster.
        """
        new_centroids = []
        for cluster_idx in range(self.num_clusters):
            curr_cluster_samples = X[self.labels == cluster_idx]
            centroid = curr_cluster_samples.mean(axis=0)
            new_centroids.append(centroid)
        return np.array(new_centroids)

    def predict(self, X):
        """
        Predicts the closest cluster for each sample in X.

        Parameters:
            X (ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            labels (ndarray): Predicted cluster labels for each sample in X.
        """
        return self._assign_clusters(X)
