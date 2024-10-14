from utils import *

def linear_regression(vars, arrays, independent_var, dependent_var):
    # Get independent and dependent variables
    independent_var_index = vars.index(independent_var)
    dependent_var_index = vars.index(dependent_var)
    
    X = [[float(arrays[independent_var_index][i])] for i in range(len(arrays[independent_var_index]))]
    Y = [float(y) for y in arrays[dependent_var_index]]
    
    # Calculate coefficients
    coefficients = calculate_coefficients(X, Y)
    
    # Calculate intercept
    intercept = calculate_intercept(X, Y, coefficients)
    
    return coefficients, intercept

vars, arrays = read_csv("test.csv")
independent_var = input("Enter independent variable: ")
dependent_var = input("Enter dependent variable: ")

coefficients, intercept = linear_regression(vars, arrays, independent_var, dependent_var)

print("Coefficients: ", coefficients)
print("Intercept: ", intercept)