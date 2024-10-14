from utils import *

vars, arrays = (read_csv("test.csv"))
arrays_bar = []

def multiple_regression(vars, arrays, arrays_bar, independent_vars, dependent_var):
    # Calculate coefficients
    coefficients = [0] * len(independent_vars)
    intercept = 0
    
    # Calculate X^T X
    XT_X = [[0 for _ in range(len(independent_vars) + 1)] for _ in range(len(independent_vars) + 1)]
    for i in range(len(arrays[0])):
        for j in range(len(independent_vars)):
            XT_X[0][j + 1] += 1
            XT_X[j + 1][0] += 1
            for k in range(len(independent_vars)):
                XT_X[j + 1][k + 1] += float(arrays[vars.index(independent_vars[j])][i]) * float(arrays[vars.index(independent_vars[k])][i])
    
    # Calculate X^T Y
    XT_Y = [0] * (len(independent_vars) + 1)
    for i in range(len(arrays[0])):
        XT_Y[0] += float(arrays[vars.index(dependent_var)][i])
        for j in range(len(independent_vars)):
            XT_Y[j + 1] += float(arrays[vars.index(independent_vars[j])][i]) * float(arrays[vars.index(dependent_var)][i])
    
    # Solve for coefficients using Gaussian elimination
    coefficients = gaussian_elimination(XT_X, XT_Y)
    
    # Calculate intercept
    intercept = arrays_bar[vars.index(dependent_var)] - sum([coefficients[j + 1] * arrays_bar[vars.index(independent_vars[j])] for j in range(len(independent_vars))])
    
    return coefficients, intercept

def gaussian_elimination(matrix, vector):
    n = len(matrix)
    
    for i in range(n):
        # Search for maximum in this column
        max_el = abs(matrix[i][i])
        max_row = i
        for k in range(i+1, n):
            if abs(matrix[k][i]) > max_el:
                max_el = abs(matrix[k][i])
                max_row = k
        
        # Swap maximum row with current row
        if max_row != i:
            matrix[i], matrix[max_row] = matrix[max_row], matrix[i]
            vector[i], vector[max_row] = vector[max_row], vector[i]
        
        # Make all rows below this one 0 in current column
        for k in range(i+1, n):
            if matrix[i][i] == 0:
                raise ValueError("Matrix is singular")
            c = -matrix[k][i]/matrix[i][i]
            for j in range(i, n):
                matrix[k][j] += c * matrix[i][j]
            vector[k] += c * vector[i]
    
    # Solve equation Ax=b for an upper triangular matrix A
    x = [0 for i in range(n)]
    for i in range(n-1, -1, -1):
        if matrix[i][i] == 0:
            raise ValueError("Matrix is singular")
        x[i] = vector[i]/matrix[i][i]
        for k in range(i-1, -1, -1):
            vector[k] -= matrix[k][i] * x[i]
    return x

# Calculate averages
for i in range(len(arrays)):
    arrays_bar.append(average_array(arrays[i]))

# Get independent and dependent variables
csv_vars = ",".join(vars)
independent_vars = input(f"Enter independent variables({csv_vars}): ").split(",")
dependent_var = input(f"Enter dependent variable({csv_vars}): ")

# Calculate coefficients and intercept
coefficients, intercept = multiple_regression(vars, arrays, arrays_bar, independent_vars, dependent_var)

# Print results
print(dependent_var, "=", coefficients[0], "+", " + ".join([f"{coefficients[j + 1]}*{independent_vars[j]}" for j in range(len(independent_vars))]), "+", intercept)