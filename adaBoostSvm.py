import numpy as np
import pandas as pd
from keras.utils import to_categorical


def one_cycle(x_train, y_train, x_test, y_test):

    def accuracy(y_pred, y_true):
        return (y_pred == y_true).mean()

    def h(w, b, x):

        return x @ w - b

    def predict(w, b, x):
        return h(w, b, x).argmax(axis=1)

    def H(x):
        return np.sign(x) / 2 + 1/2

    def grad(w, b, x, y, c):

        hep = h(w, b, x)

        dw1 = -c * ((H(1 - y.T[0] * hep.T[0]) * y.T[0])[:, np.newaxis] * x).sum(axis=0) + 2 * w.T[0]
        dw2 = -c * ((H(1 - y.T[1] * hep.T[1]) * y.T[1])[:, np.newaxis] * x).sum(axis=0) + 2 * w.T[1]

        return (
                np.stack([dw1, dw2], axis=1),
                c * (H(1 - y * hep) * y).sum(axis=0)
        )

    class SVM:
        def __init__(self, w, b):
            self.w = w
            self.b = b

        def predict(self, x):
            return predict(self.w, self.b, x)

        def predictPlusMin(self, x):
            return self.predict(x) * 2 - 1

    def trainSVM(x, y):
        w = np.random.normal(size=(8, 2))
        b = np.random.normal(size=(2,))

        c = 0.01
        lr = 0.0001

        encoded_y = to_categorical(y)

        encoded_y = (encoded_y * 2) - 1

        for i in range(500):
            dw, db = grad(w, b, x, encoded_y, c)
            w = w - lr * dw
            b = b - lr * db

        return SVM(w, b)

    def adaboost(x, y, steps=1):

        svms = list()

        _x = x
        _y = y
        y_ada = (y * 2) - 1

        for _ in range(steps):
            svm = trainSVM(_x, _y)
            y_pred = svm.predict(_x)

            y_pred = (y_pred * 2) - 1

            error = (y_ada != y_pred).mean()

            W = np.log2((1 - error) / (error + 1e-8)) / 2

            svms.append((svm, W))

            w = (1 / x.shape[0]) * np.exp(W * -(y_pred * y_ada))
            w = w / w.sum()
            idx = np.random.choice(_x.shape[0], p=w, size=(_x.shape[0]))

            _x = _x[idx]
            _y = _y[idx]
            y_ada = y_ada[idx]

        return svms

    def adaPredict(svms, x):
        y_preds = [svm.predictPlusMin(x) * W for svm, W in svms]
        y_preds = np.stack(y_preds, axis=1).sum(axis=1)
        y_preds = (np.sign(y_preds) + 1) / 2
        return y_preds

    svms = adaboost(x_train, y_train, steps=10)

    return accuracy(adaPredict(svms, x_test), y_test) * 100


df = pd.read_csv('pulsar_stars.csv')
x = df.values

y = x[:, 8]
x = x[:, :8]

result_accuracy = np.array([])

for k in range(10):
    print(k)
    x_test = x[int(k * (x.shape[0] / 10)): int((k + 1) * (x.shape[0] / 10))]
    y_test = y[int(k * (y.shape[0] / 10)): int((k + 1) * (y.shape[0] / 10))]

    x_train_part_1 = x[0: int(k * (x.shape[0] / 10))]
    x_train_part_2 = x[int((k + 1) * (x.shape[0] / 10)): x.shape[0] - 1]
    x_train = np.append(x_train_part_1, x_train_part_2, axis=0)

    y_train_part_1 = y[0: int(k * (y.shape[0] / 10))]
    y_train_part_2 = y[int((k + 1) * (y.shape[0] / 10)): y.shape[0] - 1]
    y_train = np.append(y_train_part_1, y_train_part_2, axis=0)

    result_accuracy = np.append(result_accuracy, one_cycle(x_train, y_train, x_test, y_test))


print("Resulted accuracy: %.2f" % result_accuracy.mean() + "%")
