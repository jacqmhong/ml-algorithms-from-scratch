# knn.py
import numpy as np
from collections import Counter

class KNearestNeighbors:
    def __init__(self, k=5):
        """
        Initializes the k-Nearest Neighbors classifier.

        Parameters:
            k (int): The number of neighbors to consider for classification. Odd number recommended to avoid voting ties.
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Stores the training data.

        Parameters:
            X (ndarray): Feature matrix of shape (n_samples, n_features).
            y (ndarray): Target vector of shape (n_samples,).
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predicts the class labels for the given data.

        Parameters:
            X (ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            ndarray: Predicted class labels for each sample in X.
        """
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x):
        """
        Predicts the class label for a single sample.

        Parameters:
            x (ndarray): A single data point of shape (n_features,).

        Returns:
            int: Predicted class label (the majority vote of the k nearest samples).
        """
        # Compute distances from x to all training samples
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]

        # Get the indices and labels of the k nearest samples
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Return the most common label (majority vote)
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def _euclidean_distance(self, x1, x2):
        """
        Calculates the Euclidean distance between two points.

        Parameters:
            x1, x2 (ndarray): Data points of shape (n_features,).

        Returns:
            float: Euclidean distance between x1 and x2.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))


"""
To Test Implementation:

# Sample data
X_train = np.array([[1, 2], [2, 3], [3, 3], [6, 5], [7, 8]])
y_train = np.array([0, 0, 0, 1, 1])

# Initialize and train k-NN
knn = KNearestNeighbors(k=3)
knn.fit(X_train, y_train)

# Test data
X_test = np.array([[3, 4], [5, 5]])

# Predictions
predictions = knn.predict(X_test)
print("Predictions:", predictions)
"""
