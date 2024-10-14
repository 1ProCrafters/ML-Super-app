# utils/data_processing.py
import random #inbuilt in python
import math #inbuilt in python
import csv #inbuilt in python

def load_csv(filepath):
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    header = data[0]
    rows = data[1:]
    return header, rows

def preprocess_data(rows, target_index):
    X = []
    y = []
    for row in rows:
        features = [float(value) for i, value in enumerate(row) if i != target_index]
        X.append(features)
        y.append(float(row[target_index]))
    return X, y

def normalize(X):
    min_vals = [min(feature) for feature in zip(*X)]
    max_vals = [max(feature) for feature in zip(*X)]
    normalized_X = []
    for x in X:
        normalized_x = [(xi - min_val) / (max_val - min_val) if max_val != min_val else 0 for xi, min_val, max_val in zip(x, min_vals, max_vals)]
        normalized_X.append(normalized_x)
    return normalized_X

def standardize(X):
    means = [sum(feature)/len(feature) for feature in zip(*X)]
    stds = [math.sqrt(sum((xi - mean) ** 2 for xi in feature)/len(feature)) for feature, mean in zip(zip(*X), means)]
    standardized_X = []
    for x in X:
        standardized_x = [(xi - mean) / std if std != 0 else 0 for xi, mean, std in zip(x, means, stds)]
        standardized_X.append(standardized_x)
    return standardized_X

def train_test_split(X, y, test_size=0.2):
    combined = list(zip(X, y))
    random.shuffle(combined)
    X[:], y[:] = zip(*combined)
    split = int(len(X) * (1 - test_size))
    return X[:split], X[split:], y[:split], y[split:]