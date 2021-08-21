from matplotlib.pyplot import axis
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import math


def knn(x_train, y_train, knn_k):
    no_rows = x_train.shape[0]
    # train_size = math.ceil(no_rows*0.8)
    # print("train size:", train_size)
    nn = KNeighborsClassifier(n_neighbors=knn_k)
    x_new = []
    y_new = []
    x_train = np.array(x_train)
    # print(np.array(x_train))
    for i, x in enumerate(x_train):
        # print("train:", x)
        if y_train[i] != "unjoinable":
            if len(x_new) == 0:
                x_new = np.expand_dims(x, axis=0)
            else:
                x_new = np.concatenate((x_new, [x]), axis=0)
            y_new.append(y_train[i])
    nn.fit(x_new, y_new)
    # nn.fit(x_train[:train_size], y_train[:train_size])
    y_pred_train = nn.predict(x_train)
    # y_pred = nn.predict(x_train)
    # print("------k =", knn_k)
    # print("y_pred:", y_pred)

    # ground_truth = y_train

    # # print("ground_truth:", ground_truth)
    # correct = 0
    # for i, y in enumerate(ground_truth):
    #     if y_pred[i] == y:
    #         correct += 1
    # test_acc = correct*100/len(ground_truth)

    traincorrect = 0
    for i, y in enumerate(y_train):
        if y_pred_train[i] == y:
            traincorrect += 1

    train_acc = traincorrect * 100/len(y_train)

    return train_acc, 100, nn


def predict(nn, x_test):
    return nn.predict(x_test)
