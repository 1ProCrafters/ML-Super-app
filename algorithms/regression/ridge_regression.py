# algorithms/regression/ridge_regression.py

from .utils import add_bias_term, mean_squared_error
from utils.metrics import mean_squared_error as mse_metric

class RidgeRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, alpha=1.0):
        """
        Initialize the Ridge Regression model.

        Args:
            learning_rate (float): Learning rate for gradient descent.
            epochs (int): Number of iterations for training.
            alpha (float): Regularization strength.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = alpha
        self.weights = []

    def fit(self, X, y):
        """
        Train the Ridge Regression model using Gradient Descent with L2 regularization.

        Args:
            X (list of list of floats): Feature matrix.
            y (list of floats): Target vector.
        """
        X = add_bias_term(X)
        n_samples, n_features = len(X), len(X[0])
        # Initialize weights to zeros
        self.weights = [0.0 for _ in range(n_features)]

        for epoch in range(self.epochs):
            predictions = self.predict(X)
            errors = [yp - yt for yp, yt in zip(predictions, y)]

            # Update weights with regularization (excluding bias term)
            for i in range(n_features):
                gradient = (sum(e * x[i] for e, x in zip(errors, X)) + self.alpha * self.weights[i]) / n_samples
                self.weights[i] -= self.learning_rate * gradient

            # Optionally, print the loss every 100 epochs
            if (epoch + 1) % 100 == 0:
                loss = mean_squared_error(y, predictions)
                print(f"Epoch {epoch + 1}/{self.epochs}, MSE: {loss:.4f}")

    def predict(self, X):
        """
        Predict target values using the trained Ridge Regression model.

        Args:
            X (list of list of floats): Feature matrix.

        Returns:
            list of floats: Predicted target values.
        """
        if not self.weights:
            raise ValueError("Model has not been trained yet.")

        # If X does not have bias term, add it
        if len(X[0]) + 1 == len(self.weights):
            X = add_bias_term(X)

        predictions = [sum(w * xi for w, xi in zip(self.weights, x)) for x in X]
        return predictions
