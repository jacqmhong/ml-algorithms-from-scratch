# __init__.py
# Importing each algorithm as part of the 'algorithms' package

# TODO Algorithms:
from .linear_regression import LinearRegression
# from .logistic_regression import LogisticRegression
# from .decision_tree import DecisionTree
# from .random_forest import RandomForest
from .k_means import KMeans
from .knn import KNearestNeighbors
# from .pca import PCA
# from .gradient_descent import GradientDescent
# from .naive_bayes import NaiveBayes
# from .adaboost import AdaBoost

__all__ = [
    "LinearRegression",
    # "LogisticRegression",
    # "DecisionTree",
    # "RandomForest",
    "KMeans",
    "KNearestNeighbors",
    # "PCA",
    # "GradientDescent",
    # "NaiveBayes",
    # "AdaBoost"
]