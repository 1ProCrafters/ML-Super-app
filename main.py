# main.py

import sys
import importlib
from utils.data_processing import load_csv, preprocess_data
from utils.metrics import mean_squared_error, accuracy

topic = ""
model_name = ""
dataset_path = ""
target_column = ""
params = ""

def get_model(module_path, class_name):
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    return model_class

def main():
    if len(sys.argv) < 5:
        print("Usage: python main.py <topic> <model_name> <dataset_path> <target_column> [params]")
        topic = input("Enter topic: ")
        model_name = input("Enter model name: ")
        dataset_path = input("Enter dataset path: ")
        target_column = input("Enter target column: ")
        params = input("Enter model parameters (comma-separated): ").split(",")
    else:
        topic = sys.argv[1]
        model_name = sys.argv[2]
        dataset_path = sys.argv[3]
        target_column = sys.argv[4]
        params = sys.argv[5:]

    # Load dataset
    header, rows = load_csv(dataset_path)
    target_index = header.index(target_column)
    X, y = preprocess_data(rows, target_index)

    # Dynamically load the model
    module_path = f"algorithms.{topic}.{model_name}"
    class_name = ''.join(word.capitalize() for word in model_name.split('_'))
    try:
        ModelClass = get_model(module_path, class_name)
    except (ModuleNotFoundError, AttributeError) as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Initialize model with parameters
    model_params = {}
    for param in params:
        key, value = param.split('=')
        # Attempt to convert to int or float, otherwise keep as string
        if value.isdigit():
            model_params[key] = int(value)
        else:
            try:
                model_params[key] = float(value)
            except ValueError:
                model_params[key] = value

    model = ModelClass(**model_params)

    # Train the model
    model.fit(X, y)

    # Make predictions
    predictions = model.predict(X)

    # Evaluate
    if topic == 'regression':
        mse = mean_squared_error(y, predictions)
        print(f"Mean Squared Error: {mse}")
    elif topic == 'classification':
        # For models that predict class labels
        acc = accuracy(y, predictions)
        print(f"Accuracy: {acc}")
    elif topic == 'clustering':
        # Clustering algorithms do not have ground truth labels
        # Optionally, compute metrics like silhouette score if labels are available
        print("Clustering completed. No evaluation metric available without true labels.")
    else:
        print("Evaluation metrics not defined for this topic.")

if __name__ == "__main__":
        main()
