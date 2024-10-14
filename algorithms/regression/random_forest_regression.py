# algorithms/regression/random_forest_regression.py

import random
from .decision_tree_regression import DecisionTreeRegression
from .utils import mean_squared_error

class RandomForestRegression:
    def __init__(self, n_trees=10, min_samples_split=2, max_depth=100, sample_size=1.0):
        """
        Initialize the Random Forest Regression model.

        Args:
            n_trees (int): Number of decision trees in the forest.
            min_samples_split (int): Minimum number of samples required to split a node.
            max_depth (int): Maximum depth of each tree.
            sample_size (float): Proportion of the dataset to use for training each tree.
        """
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.sample_size = sample_size
        self.trees = []

    def fit(self, X, y):
        """
        Train the Random Forest Regression model.

        Args:
            X (list of list of floats): Feature matrix.
            y (list of floats): Target vector.
        """
        self.trees = []
        n_samples = len(X)
        sample_size = int(n_samples * self.sample_size)

        for _ in range(self.n_trees):
            # Bootstrap sampling
            indices = [random.randint(0, n_samples - 1) for _ in range(sample_size)]
            sample_X = [X[i] for i in indices]
            sample_y = [y[i] for i in indices]

            # Create and train a decision tree
            tree = DecisionTreeRegression(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            tree.fit(sample_X, sample_y)
            self.trees.append(tree)

    def predict(self, X):
        """
        Predict target values using the trained Random Forest Regression model.

        Args:
            X (list of list of floats): Feature matrix.

        Returns:
            list of floats: Predicted target values.
        """
        tree_predictions = [tree.predict(X) for tree in self.trees]
        # Transpose the list to aggregate predictions for each sample
        aggregated_predictions = []
        for i in range(len(X)):
            # Calculate the mean prediction from all trees
            sample_preds = [tree_preds[i] for tree_preds in tree_predictions]
            aggregated_predictions.append(sum(sample_preds) / len(sample_preds))
        return aggregated_predictions
