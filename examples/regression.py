import numpy as np

#hours studied, practrice problems solved
X_raw = np.array([
    [6, 7],
    [2, 3],
    [4, 5],
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
    [8, 9],
    [9, 10],
    [10, 11],
    [5, 13],
    [8, 22],
    [12, 15],
    [3, 12],
    [4, 14],
    [6, 18],
    [7, 20],
    [9, 25],
    [11, 30],
    [13, 35]
])

#note in the exam
Y_raw = np.array([
    [85],
    [60],
    [70],
    [55],
    [65],
    [75],
    [80],
    [85],
    [90],
    [95],
    [88],
    [92],
    [94],
    [86],
    [89],
    [91],
    [93],
    [96],
    [98],
    [99]
])

#split into training data and test data
split = int(len(X_raw)*0.8)
X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
Y_train_raw, Y_test_raw = Y_raw[:split], Y_raw[split:]

#normalize the data
X_min, X_max = X_train_raw.min(axis=0), X_train_raw.max(axis=0)
Y_min, Y_max = Y_train_raw.min(axis=0), Y_train_raw.max(axis=0)

def normalize_X(data):
    return (data - X_min)/(X_max - X_min)

def normalize_Y(data):
    return (data - Y_min)/(Y_max - Y_min)


