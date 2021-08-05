from sklearn.neighbors import KNeighborsClassifier
import math


def knn(x_train, y_train, knn_k):
    no_rows = x_train.shape[0]
    train_size = math.ceil(no_rows*0.8)
    # print("train size:", train_size)
    nn = KNeighborsClassifier(n_neighbors=knn_k)
    nn.fit(x_train[:train_size], y_train[:train_size])
    y_pred_train = nn.predict(x_train[:train_size])
    y_pred = nn.predict(x_train[train_size:])

    # print("y_pred:", y_pred)

    ground_truth = y_train[train_size:]

    # print("ground_truth:", ground_truth)
    correct = 0
    for i, y in enumerate(ground_truth):
        if y_pred[i] == y:
            correct += 1
    test_acc = correct*100/len(ground_truth)

    traincorrect = 0
    for i, y in enumerate(y_train[:train_size]):
        if y_pred_train[i] == y:
            traincorrect += 1

    train_acc = traincorrect * 100/len(y_train[:train_size])

    return train_acc, test_acc
