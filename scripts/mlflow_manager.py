import mlflow
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_and_log_model(pipeline, params, X_train, X_test, y_train, y_test, dataset, dataset_path):
    """
    Generic model registration function for MLflow.
    
    This is a generic function to register models to MLflow. It receives the classifier, X and Y matrices, the specific parameters
    for the model (estimator), the dataset used in its training, and the path to the dataset file in the project.
    Its functionality is to train the model and perform its run registration in MLflow, in addition to registering other information such as:
    parameters, metrics (accuracy, precision, recall & f1-score), input example, model signature, and the dataset used.
    
    Args:
        pipeline: Scikit-Learn Pipeline object containing preprocessor and classifier steps.
        params (dict): Dictionary of best parameters chosen via GridSearchCV.
        X_train (numpy.ndarray): Feature matrix used for training.
        y_train (numpy.ndarray): Vector or matrix of target labels for training.
        X_test (numpy.ndarray): Feature matrix used for testing.
        y_test (numpy.ndarray): Vector or matrix of target labels for testing.
        dataset (pandas.core.frame.DataFrame): The dataset used for model training.
        dataset_path (str): Path to the dataset file in the project.
    
    Returns:
        model_info: Information returned after tracking the model in MLflow.
    """

    mlflow_dataset = mlflow.data.from_pandas(
        dataset, 
        source=dataset_path,
        name=dataset_path.split('/')[-1]
    )
    # Get the name of the estimator (last step in the pipeline)
    estimator_name = pipeline.steps[-1][1].__class__.__name__
    print('Name of the model: ' + estimator_name)

    with mlflow.start_run(run_name=estimator_name):
    
        # Model training
        # Set the best pparams as the params of the pipeline
        pipeline.set_params(**params)
        pipeline.fit(X_train, y_train)
        # Pipeline forecast
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)
    
        # Record in MLflow 
        # Parameter registration
        mlflow.log_params(params)
    
        # Key metrics registration
        mlflow.log_metric('ROC_AUC', roc_auc_score(y_test, y_pred_proba[:, 1]))
        mlflow.log_metric('Accuracy', accuracy_score(y_test, y_pred))
        mlflow.log_metric('Precision', precision_score(y_test, y_pred))
        mlflow.log_metric('Recall', recall_score(y_test, y_pred))
        mlflow.log_metric('f1 score', f1_score(y_test, y_pred))
    
        # Dataset registration
        mlflow.log_input(mlflow_dataset, context='raw-data')
    
        # Tags registration
        mlflow.set_tag('Training info', f'Basic {estimator_name} for heart disease data')
        mlflow.set_tag('Model origin', 'Generic function for training and registering model.')
    
        signature = infer_signature(X_train, pipeline.predict(X_test))
    
        # Registration of the model itself
        # The variable `model_info` contains information returned after tracking the model
        model_info = mlflow.sklearn.log_model(
            name=estimator_name,
            sk_model=pipeline,
            signature=signature, 
            input_example=X_test.iloc[[0]],
            registered_model_name=estimator_name
        )
    
        return model_info