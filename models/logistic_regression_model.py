from sklearn.linear_model import LogisticRegression

def train_logistic_regression(X_train, y_train, C=1.0):
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    return model
