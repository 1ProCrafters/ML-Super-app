def read_csv(path):
    vars = []
    arrays = []
    
    f = open(path, "r")
    lines = f.readlines()
    f.close()
    rows = []
    
    for line in lines:
        line = line.replace("\n", "")
        line = line.split(",")
        rows.append(line)
    vars = rows[0]
    rows.pop(0)
    
    for row in rows:
        for i in range(len(row)):
            if i >= len(arrays):
                arrays.append([row[i]])
            else:
                arrays[i].append(row[i])
    return vars, arrays

def average_array(array):
    sum = 0
    for i in range(len(array)):
        sum += int(array[i])
    return sum / len(array)

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

def calculate_coefficients(X, Y):
    X_transpose = transpose(X)
    X_transpose_X = multiply(X_transpose, X)
    X_transpose_Y = [dot_product(X_transpose[i], Y) for i in range(len(X_transpose))]
    coefficients = gaussian_elimination(X_transpose_X, X_transpose_Y)
    return coefficients

def calculate_intercept(X, Y, coefficients):
    sum_Y = 0
    for y in Y:
        sum_Y += int(y)
    mean_Y = sum_Y / len(Y)

    sum_X = [0 for _ in range(len(X[0]))]
    for x in X:
        for i in range(len(x)):
            sum_X[i] += int(x[i])
    mean_X = [x / len(X) for x in sum_X]

    intercept = mean_Y - dot_product(mean_X, coefficients)
    return intercept

def transpose(matrix):
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]

def multiply(matrix1, matrix2):
    if isinstance(matrix2[0], float):
        result = [0 for _ in range(len(matrix1))]
        for i in range(len(matrix1)):
            for j in range(len(matrix1[0])):
                result[i] += matrix1[i][j] * matrix2[i]
        return result
    else:
        result = [[0 for _ in range(len(matrix2[0]))] for _ in range(len(matrix1))]
        for i in range(len(matrix1)):
            for j in range(len(matrix2[0])):
                for k in range(len(matrix2)):
                    result[i][j] += matrix1[i][k] * matrix2[k][j]
        return result

def dot_product(vector1, vector2):
    result = 0
    for i in range(len(vector1)):
        result += vector1[i] * vector2[i]
    return result