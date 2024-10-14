# algorithms/classification/random_forest.py

import random
from algorithms.classification.decision_tree import DecisionTree

class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, min_size=1, sample_size=1.0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_size = min_size
        self.sample_size = sample_size
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            # Bootstrap sampling
            sample_X, sample_y = self._bootstrap_sample(X, y)
            tree = DecisionTree(max_depth=self.max_depth, min_size=self.min_size)
            tree.fit(sample_X, sample_y)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = [tree.predict(X) for tree in self.trees]
        # Transpose the list to get predictions for each sample
        tree_preds = list(zip(*tree_preds))
        # Majority vote
        return [max(set(preds), key=preds.count) for preds in tree_preds]

    def _bootstrap_sample(self, X, y):
        n_samples = int(len(X) * self.sample_size)
        indices = [random.randint(0, len(X) - 1) for _ in range(n_samples)]
        sample_X = [X[i] for i in indices]
        sample_y = [y[i] for i in indices]
        return sample_X, sample_y
