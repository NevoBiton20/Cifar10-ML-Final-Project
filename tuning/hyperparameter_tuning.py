from sklearn.model_selection import GridSearchCV

def tune_model(model_class, X_train, y_train, param_grid):
    grid = GridSearchCV(model_class(), param_grid, cv=3, scoring='accuracy')
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_
