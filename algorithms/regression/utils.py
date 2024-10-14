# algorithms/regression/utils.py

import math
import random

def normalize(X):
    min_vals = [min(feature) for feature in zip(*X)]
    max_vals = [max(feature) for feature in zip(*X)]
    normalized_X = []
    for x in X:
        normalized_x = [
            (xi - min_val) / (max_val - min_val) if max_val != min_val else 0
            for xi, min_val, max_val in zip(x, min_vals, max_vals)
        ]
        normalized_X.append(normalized_x)
    return normalized_X

def mean_squared_error(y_true, y_pred):
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)

def mean_absolute_error(y_true, y_pred):
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / len(y_true)

def train_test_split(X, y, test_size=0.2, shuffle=True):
    combined = list(zip(X, y))
    if shuffle:
        random.shuffle(combined)
    X[:], y[:] = zip(*combined)
    split_index = int(len(X) * (1 - test_size))
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

def add_bias_term(X):
    return [[1] + x for x in X]

def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))

def multiply_matrices(a, b):
    result = []
    b_t = list(zip(*b))
    for row in a:
        result_row = [dot_product(row, col) for col in b_t]
        result.append(result_row)
    return result

def transpose_matrix(matrix):
    return list(map(list, zip(*matrix)))

def invert_matrix(matrix):
    n = len(matrix)
    AM = [row[:] for row in matrix]
    I = [[float(i == j) for i in range(n)] for j in range(n)]

    for fd in range(n):
        if AM[fd][fd] == 0:
            # Find a row to swap
            for i in range(fd + 1, n):
                if AM[i][fd] != 0:
                    AM[fd], AM[i] = AM[i], AM[fd]
                    I[fd], I[i] = I[i], I[fd]
                    break
            else:
                raise ValueError("Matrix is singular and cannot be inverted.")

        # Normalize the pivot row
        pivot = AM[fd][fd]
        AM[fd] = [x / pivot for x in AM[fd]]
        I[fd] = [x / pivot for x in I[fd]]

        # Eliminate the current column in other rows
        for i in range(n):
            if i != fd:
                factor = AM[i][fd]
                AM[i] = [a - factor * b for a, b in zip(AM[i], AM[fd])]
                I[i] = [a - factor * b for a, b in zip(I[i], I[fd])]

    return I
