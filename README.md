# Machine Learning from Scratch in Python

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Python-blue)

Welcome to the **Machine Learning from Scratch** project! This repository contains implementations of various machine learning algorithms built entirely with plain Python, without relying on any external libraries. The project is organized into distinct modules covering major machine learning topics, providing a comprehensive framework for understanding and experimenting with foundational ML algorithms.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Algorithms Included](#algorithms-included)
  - [Classification](#classification)
  - [Regression](#regression)
  - [Clustering](#clustering)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Usage](#usage)
- [Data Processing](#data-processing)
- [Adding New Algorithms](#adding-new-algorithms)
- [Evaluation Metrics](#evaluation-metrics)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

This project aims to provide a hands-on learning experience by implementing key machine learning algorithms from the ground up using only Python's standard library. By building these models from scratch, you can gain a deeper understanding of their inner workings, strengths, and limitations.

## Project Structure

The project is organized into a modular directory structure, categorizing algorithms based on their primary machine learning tasks. Here's an overview of the directory layout:

```
ML-Models/
│
├── main.py
├── README.md
├── data/
│   └── ... (datasets)
│
├── algorithms/
│   ├── __init__.py
│   │
│   ├── classification/
│   │   ├── __init__.py
│   │   ├── utils.py
│   │   ├── logistic_regression.py
│   │   ├── decision_tree.py
│   │   ├── support_vector_machine.py
│   │   ├── naive_bayes.py
│   │   ├── k_nearest_neighbors.py
│   │   ├── random_forest.py
│   │   └── gradient_boosting.py
│   │
│   ├── regression/
│   │   ├── __init__.py
│   │   ├── utils.py
│   │   ├── linear_regression.py
│   │   ├── ridge_regression.py
│   │   ├── lasso_regression.py
│   │   ├── polynomial_regression.py
│   │   ├── support_vector_regression.py
│   │   ├── decision_tree_regression.py
│   │   └── random_forest_regression.py
│   │
│   └── clustering/
│       ├── __init__.py
│       ├── utils.py
│       ├── k_means.py
│       ├── hierarchical_clustering.py
│       ├── dbscan.py
│       ├── mean_shift.py
│       └── gaussian_mixture.py
│
└── utils/
    ├── __init__.py
    ├── data_processing.py
    ├── metrics.py
    └── matrix.py
```

### Description of Key Components

- **`main.py`**: The entry point of the application. Users interact with this script to specify the model type, model name, dataset, and parameters for training and evaluation.

- **`data/`**: Directory designated for storing datasets in CSV format. Organize your datasets here for easy access.

- **`algorithms/`**: Contains subdirectories for each major machine learning topic:

  - **`classification/`**: Implements classification algorithms.
  - **`regression/`**: Implements regression algorithms.
  - **`clustering/`**: Implements clustering algorithms.

  Each subdirectory includes:

  - **`utils.py`**: Common utility functions specific to the topic.
  - **Algorithm Files**: Separate Python files for each machine learning model (e.g., `logistic_regression.py`, `k_means.py`).

- **`utils/`**: Houses general utility modules not tied to a specific ML topic:

  - **`data_processing.py`**: Functions for loading and preprocessing data.
  - **`metrics.py`**: Evaluation metrics for model performance.
  - **`matrix.py`**: Basic matrix operations required for various algorithms.

## Algorithms Included

### Classification

- **Logistic Regression** (`logistic_regression.py`)
- **Decision Tree** (`decision_tree.py`)
- **Support Vector Machine** (`support_vector_machine.py`)
- **Naive Bayes** (`naive_bayes.py`)
- **K-Nearest Neighbors** (`k_nearest_neighbors.py`)
- **Random Forest** (`random_forest.py`)
- **Gradient Boosting** (`gradient_boosting.py`)

### Regression

- **Linear Regression** (`linear_regression.py`)
- **Ridge Regression** (`ridge_regression.py`)
- **Lasso Regression** (`lasso_regression.py`)
- **Polynomial Regression** (`polynomial_regression.py`)
- **Support Vector Regression** (`support_vector_regression.py`)
- **Decision Tree Regression** (`decision_tree_regression.py`)
- **Random Forest Regression** (`random_forest_regression.py`)

### Clustering

- **K-Means** (`k_means.py`)
- **Hierarchical Clustering** (`hierarchical_clustering.py`)
- **DBSCAN** (`dbscan.py`)
- **Mean Shift** (`mean_shift.py`)
- **Gaussian Mixture Models** (`gaussian_mixture.py`)

## Getting Started

### Prerequisites

- **Python 3.6 or higher**: Ensure you have Python installed on your machine. You can download it from [Python's official website](https://www.python.org/downloads/).

### Usage

1. **Clone the Repository**

   ```bash
   git clone https://github.com/1ProCrafters/ML-Super-app/tree/ML-Models.git
   cd ML-Super-app
   ```

2. **Prepare Your Dataset**

   - Place your dataset CSV files inside the `data/` directory.
   - Ensure that the first row contains the header with column names.

3. **Run a Model**

   The `main.py` script allows you to train and evaluate any supported machine learning model. The general usage is:

   ```bash
   python main.py <topic> <model_name> <dataset_path> <target_column> [params]
   ```

   - **`<topic>`**: The machine learning category (e.g., `classification`, `regression`, `clustering`).
   - **`<model_name>`**: The specific model to use (e.g., `logistic_regression`, `k_means`).
   - **`<dataset_path>`**: Path to your dataset CSV file (e.g., `data/iris.csv`).
   - **`<target_column>`**: The name of the target variable column in your dataset.
   - **`[params]`**: Optional model-specific parameters in `key=value` format.

   ### Examples

   - **Training a Logistic Regression Classifier**

     ```bash
     python main.py classification logistic_regression data/iris.csv species learning_rate=0.01 epochs=1000
     ```

   - **Clustering with K-Means**

     ```bash
     python main.py clustering k_means data/clusters.csv feature1 feature2 k=3 max_iterations=100
     ```

   - **Training a Random Forest Regressor**

     ```bash
     python main.py regression random_forest_regression data/housing.csv price n_trees=10 max_depth=5
     ```

## Data Processing

All data handling and preprocessing are managed through utility functions located in the `utils/data_processing.py` module. Key functionalities include:

- **Loading CSV Files**

  ```python
  from utils.data_processing import load_csv

  header, rows = load_csv('data/your_dataset.csv')
  ```

- **Preprocessing Data**

  Convert raw data into feature matrices and target vectors.

  ```python
  from utils.data_processing import preprocess_data

  X, y = preprocess_data(rows, target_index)
  ```

- **Normalization and Standardization**

  Normalize or standardize features to improve model performance.

  ```python
  from utils.data_processing import normalize, standardize

  X_normalized = normalize(X)
  X_standardized = standardize(X)
  ```

- **Train-Test Split**

  Split data into training and testing sets.

  ```python
  from utils.data_processing import train_test_split

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
  ```

## Adding New Algorithms

To extend the project with new machine learning algorithms, follow these steps:

1. **Choose the Appropriate Category**

   Determine whether your algorithm falls under `classification`, `regression`, or `clustering`.

2. **Create a New Python File**

   Inside the relevant subdirectory (e.g., `algorithms/classification/`), create a new `.py` file named after your algorithm (e.g., `my_new_algorithm.py`).

3. **Implement the Model Class**

   - Ensure your class has a consistent interface with `fit` and `predict` methods.
   - Utilize utility functions from `utils/` as needed.

   ```python
   # algorithms/classification/my_new_algorithm.py

   class MyNewAlgorithm:
       def __init__(self, param1=value1, param2=value2):
           # Initialize parameters
           pass

       def fit(self, X, y):
           # Implement training logic
           pass

       def predict(self, X):
           # Implement prediction logic
           return predictions
   ```

4. **Update `main.py`**

   No changes are necessary if your algorithm follows the standard interface. You can now use it via `main.py` with the appropriate parameters.

5. **Test Your Implementation**

   Ensure your new algorithm works correctly by running it on sample datasets and verifying the results.

## Evaluation Metrics

Performance evaluation is crucial for assessing the effectiveness of your models. The `utils/metrics.py` module provides essential metrics:

- **Regression Metrics:**

  - **Mean Squared Error (MSE)**
  - **Mean Absolute Error (MAE)** _(To be implemented)_

- **Classification Metrics:**

  - **Accuracy**
  - **Precision** _(To be implemented)_
  - **Recall** _(To be implemented)_
  - **F1-Score** _(To be implemented)_

- **Clustering Metrics:**
  - **Silhouette Score** _(To be implemented)_
  - **Davies-Bouldin Index** _(To be implemented)_

### Usage Example

```python
from utils.metrics import mean_squared_error, accuracy

# For Regression
mse = mean_squared_error(y_true, y_pred)
print(f"Mean Squared Error: {mse}")

# For Classification
acc = accuracy(y_true, y_pred)
print(f"Accuracy: {acc}")
```

## Contributing

Contributions are welcome! Whether it's adding new algorithms, improving existing implementations, or enhancing documentation, your efforts are appreciated.

1. **Fork the Repository**

2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add Your Feature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- Inspired by the desire to understand machine learning algorithms at a fundamental level.
- Special thanks to the open-source community for providing valuable resources and insights.

---

Feel free to explore, learn, and contribute! If you have any questions or need assistance, please open an issue or reach out to me anytime.
