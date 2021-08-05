from sklearn.svm import SVC


def svc(x_train, y_train):
    # Support Vector Machine
    svm = SVC(kernel="rbf", C=0.025, random_state=101,
              decision_function_shape='ovo')
    svm.fit(x_train[:800], y_train[:800])
    y_pred = svm.predict(x_train[800:])
    print(y_pred)
