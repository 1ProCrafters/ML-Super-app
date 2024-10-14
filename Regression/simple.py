from utils import *

vars, arrays = (read_csv("test.csv"))
arrays_bar = []

# Calculate averages
for i in range(len(arrays)):
    arrays_bar.append(average_array(arrays[i]))

# Calculate slope and intercept
def predict(array, independent_var, dependent_var):
    dependent_var_index = vars.index(dependent_var)
    independent_var_index = vars.index(independent_var)
    
    slope = 0
    intercept = 0
    
    # Calculate slope numerator
    slope_num = 0
    for i in range(len(arrays[independent_var_index])):
        slope_num += (float(arrays[independent_var_index][i]) - float(arrays_bar[independent_var_index])) * (float(arrays[dependent_var_index][i]) - float(arrays_bar[dependent_var_index]))
    
    # Calculate slope denominator 
    slope_denom = 0
    for i in range(len(arrays[independent_var_index])):
        slope_denom += pow(float(arrays[independent_var_index][i]) - float(arrays_bar[independent_var_index]), 2)
    
    # Calculate slope
    slope = slope_num / slope_denom
    
    # Calculate intercept
    intercept = arrays_bar[dependent_var_index] - slope * arrays_bar[independent_var_index]
    
    return slope, intercept

    csv_vars = ",".join(vars)
    independent_var = input(f"Enter independent variable({csv_vars}): ")
    dependent_var = input(f"Enter dependent variable({csv_vars}): ")

    slope, intercept = predict(vars, independent_var, dependent_var)
    print(dependent_var, "=", slope, independent_var, "+", intercept)