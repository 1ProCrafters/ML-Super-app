# main.py

import sys
import importlib
import argparse
from utils.data_processing import load_csv, preprocess_data, normalize, standardize, train_test_split
from utils.metrics import mean_squared_error, accuracy, mean_absolute_error, precision, recall, f1_score, confusion_matrix

def get_model(module_path, class_name):
    """
    Dynamically import and return the specified model class.

    Args:
        module_path (str): The dot-separated path to the module.
        class_name (str): The name of the class to import.

    Returns:
        class: The model class.
    """
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    return model_class

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Machine Learning Models from Scratch in Python")

    parser.add_argument('topic', type=str, choices=['classification', 'regression', 'clustering'],
                        help='The machine learning topic: classification, regression, or clustering.')
    parser.add_argument('model_name', type=str,
                        help='The name of the model to use (e.g., logistic_regression, k_means).')
    parser.add_argument('dataset_path', type=str,
                        help='Path to the dataset CSV file (e.g., data/iris.csv).')
    parser.add_argument('--target_column', type=str, default=None,
                        help='The name of the target variable column (required for classification and regression).')
    parser.add_argument('--params', nargs='*', default=[],
                        help='Optional model parameters in key=value format (e.g., learning_rate=0.01 epochs=1000).')

    return parser.parse_args()

def parse_parameters(param_list):
    """
    Parse a list of key=value strings into a dictionary.

    Args:
        param_list (list of str): List of parameters in key=value format.

    Returns:
        dict: Dictionary of parameters with appropriate types.
    """
    params = {}
    for param in param_list:
        if '=' not in param:
            print(f"Invalid parameter format: {param}. Expected key=value.")
            sys.exit(1)
        key, value = param.split('=', 1)
        # Attempt to convert to int or float
        try:
            if '.' in value:
                value_converted = float(value)
            else:
                value_converted = int(value)
        except ValueError:
            # Keep as string if not a number
            value_converted = value
        params[key] = value_converted
    return params

def main():
    args = parse_arguments()

    topic = args.topic.lower()
    model_name = args.model_name.lower()
    dataset_path = args.dataset_path
    target_column = args.target_column
    params = parse_parameters(args.params)

    # Validate target_column for classification and regression
    if topic in ['classification', 'regression'] and not target_column:
        print(f"The --target_column argument is required for {topic} tasks.")
        sys.exit(1)
    if topic == 'clustering' and target_column:
        print("Warning: --target_column argument is ignored for clustering tasks.")

    # Load dataset
    header, rows = load_csv(dataset_path)

    if topic in ['classification', 'regression']:
        if target_column not in header:
            print(f"Target column '{target_column}' not found in the dataset.")
            sys.exit(1)
        target_index = header.index(target_column)
        X, y = preprocess_data(rows, target_index)
    elif topic == 'clustering':
        # For clustering, use all columns as features
        X = [[float(value) for value in row] for row in rows]
        y = None  # No target variable

    # Preprocess data (optional: normalize or standardize)
    # X = normalize(X)
    # X = standardize(X)

    # Optionally, split data into train and test sets for classification and regression
    if topic in ['classification', 'regression']:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    else:
        X_train, X_test, y_train, y_test = X, X, None, None

    # Dynamically load the model
    module_path = f"algorithms.{topic}.{model_name}"
    class_name = ''.join(word.capitalize() for word in model_name.split('_'))
    try:
        ModelClass = get_model(module_path, class_name)
    except (ModuleNotFoundError, AttributeError) as e:
        print(f"Error loading model '{model_name}' from topic '{topic}': {e}")
        sys.exit(1)

    # Initialize model with parameters
    try:
        model = ModelClass(**params)
    except TypeError as e:
        print(f"Error initializing model with parameters: {e}")
        sys.exit(1)

    # Train the model
    print(f"Training {class_name}...")
    if topic in ['classification', 'regression']:
        model.fit(X_train, y_train)
    elif topic == 'clustering':
        model.fit(X_train)
    print("Training completed.")

    # Make predictions
    print("Making predictions...")
    if topic in ['classification', 'regression']:
        predictions = model.predict(X_test)
    elif topic == 'clustering':
        predictions = model.predict(X_test)
    print("Predictions completed.")

    # Evaluate the model
    print("Evaluating the model...")
    if topic == 'regression':
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
    elif topic == 'classification':
        acc = accuracy(y_test, predictions)
        prec = precision(y_test, predictions)
        rec = recall(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("Confusion Matrix:")
        for row in cm:
            print(row)
    elif topic == 'clustering':
        print("Clustering completed. No ground truth labels provided for evaluation.")
        # Optionally, implement clustering evaluation metrics if labels are available
    else:
        print("Unknown topic. No evaluation performed.")

if __name__ == "__main__":
    main()
