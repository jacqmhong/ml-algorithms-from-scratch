import numpy as np

def euclidean_distance(x1, x2):
    """
    Calculates the Euclidean distance between two points.

    Parameters:
        x1, x2 (ndarray): Data points of shape (n_features,).

    Returns:
        float: Euclidean distance between x1 and x2.
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))
