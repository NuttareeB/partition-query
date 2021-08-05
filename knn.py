from sklearn.neighbors import KNeighborsClassifier

train_size = 800


def knn(x_train, y_train):
    nn = KNeighborsClassifier(n_neighbors=3)
    nn.fit(x_train[:train_size], y_train[:train_size])
    y_pred_train = nn.predict(x_train[:train_size])
    y_pred = nn.predict(x_train[train_size:])

    print(y_pred)

    ground_truth = y_train[train_size:]

    correct = 0
    for i, y in enumerate(ground_truth):
        if y_pred[i] == y:
            correct += 1

    print("test accuracy:", str(correct*100/len(ground_truth)), "%")

    traincorrect = 0
    for i, y in enumerate(y_train[:train_size]):
        if y_pred_train[i] == y:
            traincorrect += 1

    print("train accuracy:", str(traincorrect *
          100/len(y_train[:train_size])), "%")
