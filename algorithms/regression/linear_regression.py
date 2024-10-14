# algorithms/regression/linear_regression.py

from utils.data_processing import preprocess_data
from utils.metrics import mean_squared_error

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.bias = 0

    def fit(self, X, y):
        num_features = len(X[0])
        self.weights = [0.0 for _ in range(num_features)]
        self.bias = 0.0

        for epoch in range(self.epochs):
            y_pred = self.predict(X)
            error = [yp - yt for yp, yt in zip(y_pred, y)]
            
            # Update weights and bias
            for i in range(num_features):
                gradient = sum(e * x[i] for e, x in zip(error, X)) / len(X)
                self.weights[i] -= self.learning_rate * gradient
            self.bias -= self.learning_rate * sum(error) / len(X)

            if epoch % (self.epochs / 10) == 0:
                mse = mean_squared_error(y, y_pred)
                print(f'Epoch {epoch}, MSE: {mse}')

    def predict(self, X):
        predictions = []
        for x in X:
            pred = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
            predictions.append(pred)
        return predictions
