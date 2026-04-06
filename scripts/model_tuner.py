from sklearn.model_selection import GridSearchCV

# Generic Cross Validation Function to Pipelines
def generic_grid_search(pipeline, params, X_matrix, y_matrix):
    
    grid = GridSearchCV(pipeline, param_grid=params, cv=5, scoring='roc_auc')
    grid.fit(X_matrix, y_matrix)
    best_params = grid.best_params_
    # best_estimator = grid.best_estimator_
    estimator_name = pipeline.steps[-1][1].__class__.__name__
    
    return best_params, estimator_name