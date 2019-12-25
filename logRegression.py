import numpy as np
import pandas as pd
from keras.utils import to_categorical
import common

range10000 = range(10000)

def accuracy(y_pred, y_true):
    return (y_pred == y_true).mean()

def h(w, b, x):
    return x @ w - b

def predict(w, b, x):
    return h(w, b, x).argmax(axis=1)

def H(x):
    return np.sign(x) / 2 + 1 / 2

def grad(w, b, x, y, c):
    hep = h(w, b, x)

    dw1 = -c * ((H(1 - y.T[0] * hep.T[0]) * y.T[0])[:, np.newaxis] * x).sum(axis=0) + 2 * w.T[0]
    dw2 = -c * ((H(1 - y.T[1] * hep.T[1]) * y.T[1])[:, np.newaxis] * x).sum(axis=0) + 2 * w.T[1]

    return (
        np.stack([dw1, dw2], axis=1),
        c * (H(1 - y * hep) * y).sum(axis=0)
    )

def iteration(x_train, y_train, x_test, y_test):
    encoded_y_train = to_categorical(y_train)

    w = np.zeros(shape=(8, 2))
    b = np.zeros(2)

    c = 0.01
    lr = 0.0001

    for i in range(encoded_y_train.T[0].size):
        for j in range(encoded_y_train[0].size):
            if encoded_y_train[i][j] == 0:
                encoded_y_train[i][j] = -1

    for i in range10000:
        dw, db = grad(w, b, x_train, encoded_y_train, c)
        w = w - lr * dw
        b = b - lr * db

    return accuracy(predict(w, b, x_test), y_test) * 100

def execute():
    (x, y) = common.getDataSet()
    piece10 = int(x.shape[0] / 10)
    result_accuracy = np.array([])

    for k in range(10):
        print(k)
        start = int(k * piece10)
        x_test = x[start: start + piece10]
        y_test = y[start: start + piece10]

        x_train_part_1 = x[0: start]
        x_train_part_2 = x[start + piece10: x.shape[0] - 1]
        x_train = np.append(x_train_part_1, x_train_part_2, axis=0)

        y_train_part_1 = y[0: start]
        y_train_part_2 = y[start + piece10: y.shape[0] - 1]
        y_train = np.append(y_train_part_1, y_train_part_2, axis=0)

        result_accuracy = np.append(result_accuracy, iteration(x_train, y_train, x_test, y_test))

    return result_accuracy.mean()

def output():
    print("Resulted accuracy: %.2f" % execute() + "%")

output()