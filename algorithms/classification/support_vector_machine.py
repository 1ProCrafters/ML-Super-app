# algorithms/classification/support_vector_machine.py

class SupportVectorMachine:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = 0

    def fit(self, X, y):
        n_samples, n_features = len(X), len(X[0])
        # Initialize weights
        self.w = [0.0 for _ in range(n_features)]
        self.b = 0.0

        # Convert labels to -1 and 1
        y_ = [1 if label == 1 else -1 for label in y]

        for _ in range(self.n_iters):
            for idx, x in enumerate(X):
                condition = y_[idx] * (self._dot_product(self.w, x) + self.b) >= 1
                if condition:
                    # Update weights
                    self.w = [w_i - self.lr * (2 * self.lambda_param * w_i) for w_i in self.w]
                else:
                    # Update weights and bias
                    self.w = [w_i - self.lr * (2 * self.lambda_param * w_i - y_[idx] * x_i) for w_i, x_i in zip(self.w, x)]
                    self.b += self.lr * y_[idx]

    def predict(self, X):
        linear_output = [self._dot_product(self.w, x) + self.b for x in X]
        return [1 if x >= 0 else 0 for x in linear_output]

    def _dot_product(self, a, b):
        return sum(x * y for x, y in zip(a, b))
