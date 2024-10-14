# algorithms/regression/decision_tree_regression.py

from .utils import mean_squared_error

class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        """
        Initialize a Decision Tree node.

        Args:
            feature_index (int): Index of the feature to split on.
            threshold (float): Threshold value for the split.
            left (DecisionTreeNode): Left child node.
            right (DecisionTreeNode): Right child node.
            value (float): Predicted value at the leaf node.
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeRegression:
    def __init__(self, min_samples_split=2, max_depth=100):
        """
        Initialize the Decision Tree Regression model.

        Args:
            min_samples_split (int): Minimum number of samples required to split a node.
            max_depth (int): Maximum depth of the tree.
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        """
        Train the Decision Tree Regression model.

        Args:
            X (list of list of floats): Feature matrix.
            y (list of floats): Target vector.
        """
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.

        Args:
            X (list of list of floats): Feature matrix.
            y (list of floats): Target vector.
            depth (int): Current depth of the tree.

        Returns:
            DecisionTreeNode: Root node of the subtree.
        """
        num_samples, num_features = len(X), len(X[0])

        if (num_samples < self.min_samples_split) or (depth >= self.max_depth):
            leaf_value = self._calculate_leaf_value(y)
            return DecisionTreeNode(value=leaf_value)

        best_split = self._get_best_split(X, y, num_features)
        if not best_split:
            leaf_value = self._calculate_leaf_value(y)
            return DecisionTreeNode(value=leaf_value)

        left_subtree = self._build_tree(best_split['left_X'], best_split['left_y'], depth + 1)
        right_subtree = self._build_tree(best_split['right_X'], best_split['right_y'], depth + 1)
        return DecisionTreeNode(
            feature_index=best_split['feature_index'],
            threshold=best_split['threshold'],
            left=left_subtree,
            right=right_subtree
        )

    def _get_best_split(self, X, y, num_features):
        """
        Find the best feature and threshold to split on.

        Args:
            X (list of list of floats): Feature matrix.
            y (list of floats): Target vector.
            num_features (int): Number of features.

        Returns:
            dict: Details of the best split.
        """
        best_split = {}
        min_mse = float('inf')

        for feature_index in range(num_features):
            feature_values = [x[feature_index] for x in X]
            thresholds = sorted(set(feature_values))
            for threshold in thresholds:
                left_X, left_y, right_X, right_y = self._split(X, y, feature_index, threshold)
                if len(left_y) == 0 or len(right_y) == 0:
                    continue
                current_mse = self._calculate_mse(left_y, right_y)
                if current_mse < min_mse:
                    min_mse = current_mse
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'left_X': left_X,
                        'left_y': left_y,
                        'right_X': right_X,
                        'right_y': right_y
                    }

        if min_mse == float('inf'):
            return None
        return best_split

    def _split(self, X, y, feature_index, threshold):
        """
        Split the dataset based on the feature index and threshold.

        Args:
            X (list of list of floats): Feature matrix.
            y (list of floats): Target vector.
            feature_index (int): Feature index to split on.
            threshold (float): Threshold value.

        Returns:
            tuple: left_X, left_y, right_X, right_y
        """
        left_X, left_y, right_X, right_y = [], [], [], []
        for xi, yi in zip(X, y):
            if xi[feature_index] <= threshold:
                left_X.append(xi)
                left_y.append(yi)
            else:
                right_X.append(xi)
                right_y.append(yi)
        return left_X, left_y, right_X, right_y

    def _calculate_mse(self, left_y, right_y):
        """
        Calculate the combined MSE for a split.

        Args:
            left_y (list of floats): Target values for the left split.
            right_y (list of floats): Target values for the right split.

        Returns:
            float: Combined Mean Squared Error.
        """
        total = 0
        total += mean_squared_error(left_y, [self._calculate_leaf_value(left_y)] * len(left_y))
        total += mean_squared_error(right_y, [self._calculate_leaf_value(right_y)] * len(right_y))
        return total

    def _calculate_leaf_value(self, y):
        """
        Calculate the value to store in a leaf node (mean of target values).

        Args:
            y (list of floats): Target vector.

        Returns:
            float: Mean value.
        """
        return sum(y) / len(y)

    def predict(self, X):
        """
        Predict target values using the trained Decision Tree Regression model.

        Args:
            X (list of list of floats): Feature matrix.

        Returns:
            list of floats: Predicted target values.
        """
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return predictions

    def _traverse_tree(self, x, node):
        """
        Traverse the tree to make a prediction for a single sample.

        Args:
            x (list of floats): Single feature vector.
            node (DecisionTreeNode): Current node in the tree.

        Returns:
            float: Predicted target value.
        """
        if node.value is not None:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
