# utils/matrix.py

def dot_product(a, b):
    return sum(x*y for x, y in zip(a, b))

def transpose(matrix):
    return list(map(list, zip(*matrix)))

def matrix_multiply(a, b):
    result = []
    b_t = transpose(b)
    for row in a:
        result_row = []
        for col in b_t:
            result_row.append(dot_product(row, col))
        result.append(result_row)
    return result
