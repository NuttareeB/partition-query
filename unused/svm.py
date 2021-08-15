from sklearn.svm import SVC
import math


def svc(x_train, y_train):

    no_rows = x_train.shape[0]
    train_size = math.ceil(no_rows*0.8)

    # Support Vector Machine
    svm = SVC(kernel="rbf", C=0.025, random_state=101,
              decision_function_shape='ovo')
    svm.fit(x_train[:train_size], y_train[:train_size])
    y_pred_train = svm.predict(x_train[:train_size])
    y_pred = svm.predict(x_train[train_size:])

    print("y_pred:", y_pred)

    ground_truth = y_train[train_size:]

    print("ground_truth:", ground_truth)

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

    return train_acc, test_acc, svm


def predict(svm, x_test):
    return svm.predict(x_test)
