import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iter=1000):
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.weights = None
        self.intercept = None

    def fit(self, X, y):
        """
        Uses gradient descent to fit the data to the logistic regression model.

        Parameters:
            X (ndarray): Feature matrix of shape (n_samples, n_features).
            y (ndarray): Target vector of shape (n_samples,), with values 0 or 1.
        """
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.intercept = 0

        for _ in range(self.num_iter):
            # Linear combo of x_i and w_i
            linear_combo = np.dot(X, self.weights) + self.intercept
            y_pred = 1 / (1 + np.exp(-linear_combo)) # sigmoid func to map values between 0 and 1

            # Compute the gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y)) # weight gradient
            db = (1 / num_samples) * np.sum(y_pred - y) # bias gradient

            # Update params: theta_j = theta_j - alpha * gradient
            self.weights -= self.learning_rate * dw # update weights
            self.intercept -= self.learning_rate * db # update intercept

    def predict_proba(self, X):
        """
        Predicts the probabilities for each sample belonging to the positive class.

        Parameters:
            X (ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            ndarray: Predicted probabilities for each sample.
        """
        linear_combo = np.dot(X, self.weights) + self.intercept
        y_pred = 1 / (1 + np.exp(-linear_combo))
        return y_pred

    def predict(self, X):
        """
        Predicts each sample's class label.

        Parameters:
            X (ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            ndarray: Predicted class labels (0 or 1) for each sample.
        """
        y_proba = self.predict_proba(X)
        return np.where(y_proba >= 0.5, 1, 0)
