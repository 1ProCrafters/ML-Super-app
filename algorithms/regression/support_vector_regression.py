# algorithms/regression/support_vector_regression.py

from .utils import add_bias_term, mean_squared_error
from utils.metrics import mean_squared_error as mse_metric

class SupportVectorRegression:
    def __init__(self, learning_rate=0.001, epochs=1000, lambda_param=0.01, epsilon=0.1):
        """
        Initialize the Support Vector Regression model.

        Args:
            learning_rate (float): Learning rate for gradient descent.
            epochs (int): Number of iterations for training.
            lambda_param (float): Regularization parameter.
            epsilon (float): Epsilon parameter for the epsilon-insensitive loss function.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_param = lambda_param
        self.epsilon = epsilon
        self.weights = []
        self.bias = 0.0

    def fit(self, X, y):
        """
        Train the SVR model using Gradient Descent.

        Args:
            X (list of list of floats): Feature matrix.
            y (list of floats): Target vector.
        """
        n_samples, n_features = len(X), len(X[0])
        X = add_bias_term(X)
        self.weights = [0.0 for _ in range(n_features)]
        self.bias = 0.0

        for epoch in range(self.epochs):
            for idx in range(n_samples):
                xi = X[idx]
                yi = y[idx]
                prediction = sum(w * x for w, x in zip(self.weights, xi)) + self.bias
                error = yi - prediction

                if abs(error) > self.epsilon:
                    # Update weights and bias
                    for i in range(n_features):
                        self.weights[i] += self.learning_rate * (error * xi[i] - self.lambda_param * self.weights[i])
                    self.bias += self.learning_rate * error
                else:
                    # Apply regularization
                    for i in range(n_features):
                        self.weights[i] -= self.learning_rate * self.lambda_param * self.weights[i]

            # Optionally, print the loss every 100 epochs
            if (epoch + 1) % 100 == 0:
                predictions = self.predict(X)
                loss = mean_squared_error(y, predictions)
                print(f"Epoch {epoch + 1}/{self.epochs}, MSE: {loss:.4f}")

    def predict(self, X):
        """
        Predict target values using the trained SVR model.

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

        predictions = [sum(w * xi for w, xi in zip(self.weights, x)) + self.bias for x in X]
        return predictions
