# algorithms/classification/naive_bayes.py

from math import log, pi, exp, sqrt
from collections import defaultdict

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}

    def fit(self, X, y):
        self.classes = set(y)
        for c in self.classes:
            X_c = [x for x, label in zip(X, y) if label == c]
            self.mean[c] = [sum(feature)/len(feature) for feature in zip(*X_c)]
            self.var[c] = [sum((xi - m) ** 2 for xi in feature)/len(feature) for feature, m in zip(zip(*X_c), self.mean[c])]
            self.priors[c] = len(X_c) / len(X)

    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for c in self.classes:
                prior = log(self.priors[c])
                conditional = sum(self._log_gaussian(xi, self.mean[c][i], self.var[c][i]) for i, xi in enumerate(x))
                posterior = prior + conditional
                posteriors.append(posterior)
            predictions.append(self.classes[list(posteriors).index(max(posteriors))])
        return predictions

    def _log_gaussian(self, x, mean, var):
        exponent = - ((x - mean) ** 2) / (2 * var)
        return exponent - 0.5 * log(2 * pi * var)
