# utils/metrics.py

def mean_squared_error(y_true, y_pred):
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)

def mean_absolute_error(y_true, y_pred):
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / len(y_true)

def accuracy(y_true, y_pred):
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return correct / len(y_true)

def precision(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != yp and yp == 1)
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)

def recall(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt != yp and yp == 0)
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    if prec + rec == 0:
        return 0
    return 2 * (prec * rec) / (prec + rec)

def confusion_matrix(y_true, y_pred):
    unique_classes = sorted(list(set(y_true)))
    matrix = {cls: {cls_pred:0 for cls_pred in unique_classes} for cls in unique_classes}
    for yt, yp in zip(y_true, y_pred):
        matrix[yt][yp] += 1
    # Convert dictionary to list of lists
    cm = []
    header = [""] + unique_classes
    cm.append(header)
    for cls in unique_classes:
        row = [cls] + [matrix[cls][pred_cls] for pred_cls in unique_classes]
        cm.append(row)
    return cm
