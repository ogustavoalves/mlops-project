from sklearn.model_selection import GridSearchCV

def generic_grid_search(pipeline, params, X_matrix, y_matrix):
    """
    Generic wrapper for GridSearchCV adapted to work with sklearn Pipeline objects.
    Performs hyperparameter tuning via cross-validation using ROC_AUC as the scoring metric.

    Args:
        pipeline: sklearn Pipeline object containing preprocessor and classifier steps.
        params (dict): Parameter grid for GridSearchCV containing hyperparameters to tune.
        X_matrix: Feature matrix for training.
        y_matrix: Target vector for training.

    Returns:
        tuple: Contains best_params (dict of optimal hyperparameters) and estimator_name (str).
    """
    
    grid = GridSearchCV(pipeline, param_grid=params, cv=5, scoring='roc_auc')
    grid.fit(X_matrix, y_matrix)

    best_params = grid.best_params_
    estimator_name = pipeline.steps[-1][1].__class__.__name__
    
    return best_params, estimator_name