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

def read_lines(path):
    with open(path, "rb") as f:
        num_lines = sum(1 for _ in f)
    return num_lines

def average_array(array):
    sum = 0
    for i in range(len(array)):
        sum += int(array[i])
    return sum / len(array)