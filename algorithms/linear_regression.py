import numpy as np

class LinearRegression:
    def __init__(self):
        self.weights = None
        self.intercept = None

    def fit(self, X, y):
        """
        Fits the linear regression model to the data using the OLS method.

        Parameters:
            X (ndarray): Feature matrix of shape (n_samples, n_features).
            y (ndarray): Target vector of shape (n_samples,).
        """
        # Add an intercept column to X
        X = np.column_stack((np.ones((X.shape[0])), X))

        # Calculate weights with the OLS formula:  Beta = (X_transpose * X)^-1 * X_transpose * y
        theta_best = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

        self.intercept = theta_best[0]
        self.weights = theta_best[1:]

    def predict(self, X):
        """
        Makes predictions using linear regression.

        Parameters:
            X (ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            y_pred (ndarray): Predicted target values.
        """
        # y_hat = X * weights + intercept
        return X.dot(self.weights) + self.intercept
