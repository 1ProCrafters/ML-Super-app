# algorithms/regression/polynomial_regression.py

from .linear_regression import LinearRegression
from .utils import mean_squared_error
from utils.metrics import mean_squared_error as mse_metric

class PolynomialRegression:
    def __init__(self, degree=2, learning_rate=0.01, epochs=1000):
        """
        Initialize the Polynomial Regression model.

        Args:
            degree (int): Degree of the polynomial features.
            learning_rate (float): Learning rate for gradient descent.
            epochs (int): Number of iterations for training.
        """
        self.degree = degree
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = LinearRegression(learning_rate=learning_rate, epochs=epochs)

    def _polynomial_features(self, X):
        """
        Generate polynomial features up to the specified degree.

        Args:
            X (list of list of floats): Original feature matrix.

        Returns:
            list of list of floats: Expanded feature matrix with polynomial features.
        """
        poly_X = []
        for x in X:
            poly_row = []
            for feature in x:
                for d in range(1, self.degree + 1):
                    poly_row.append(feature ** d)
            poly_X.append(poly_row)
        return poly_X

    def fit(self, X, y):
        """
        Train the Polynomial Regression model.

        Args:
            X (list of list of floats): Original feature matrix.
            y (list of floats): Target vector.
        """
        poly_X = self._polynomial_features(X)
        self.model.fit(poly_X, y)

    def predict(self, X):
        """
        Predict target values using the trained Polynomial Regression model.

        Args:
            X (list of list of floats): Original feature matrix.

        Returns:
            list of floats: Predicted target values.
        """
        poly_X = self._polynomial_features(X)
        return self.model.predict(poly_X)
