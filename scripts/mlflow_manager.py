import mlflow
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_and_log_model(pipeline, params, X_train, X_test, y_train, y_test, dataset, dataset_path):
    """
    Função "genérica" de registro de modelos no MLflow.
    
    Esta é uma função genérica para registrar os modelos ao MLflow. Ela recebe o classificador, as matrizes X e Y, os parâmetros
    especificos do modelo (estimador), o dataset usado em seu treinamento e o caminho do dataset arquivo do dataset no projeto. 
    Sua funcionalidade se dá em treinar o modelo e fazer seu registro de sua run no MLflow, além de registrar outras informações como:
    parâmetros, métricas (accuracy, precision, recall & f1-score), exemplo de input, assinatura do modelo e o dataset usado.
    
    Args:
        estimator: Classificador do Scikit-Learn.
        X_train (numpy.ndarray): Matriz de features usada para o treinamento.
        y_train (numpy.ndarray): Vetor ou matriz de rótulos alvo para treinamento.
        X_test (numpy.ndarray): Matriz de features usada para o teste.
        y_test (numpy.ndarray): Vetor ou matriz de rótulos alvo para teste.
        params (dict): Dicionário de melhores parâmetros escolhidos via GridSearchCV.
        dataset (pandas.core.frame.DataFrame): 
    
    Returns:
        
    """
    mlflow_dataset = mlflow.data.from_pandas(
        dataset, 
        source=dataset_path,
        name=dataset_path.split('/')[-1]
    )
    # Obter nome do modelo no step final da pipeline
    estimator_name = pipeline.steps[-1][1].__class__.__name__
    print('Model name: ' + estimator_name)
    
    # Wrapper de run
    with mlflow.start_run(run_name=estimator_name):
        
        # Treinamento do modelo
        # Definir hiperparâmetros na pipeline
        pipeline.set_params(**params)
        # Fitagem dos dados no algoritmo
        pipeline.fit(X_train, y_train)
        # Inferência do modelo
        y_pred = pipeline.predict(X_test)
        # predict proba para cálculo de roc auc score
        y_pred_proba = pipeline.predict_proba(X_test)
        
        # Registro no MLflow
        # Registro dos parâmetros
        mlflow.log_params(params)
        
        # Registro das métricas principais
        mlflow.log_metric('ROC_AUC', roc_auc_score(y_test, y_pred_proba[:, 1]))
        mlflow.log_metric('Accuracy', accuracy_score(y_test, y_pred))
        mlflow.log_metric('Precision', precision_score(y_test, y_pred))
        mlflow.log_metric('Recall', recall_score(y_test, y_pred))
        mlflow.log_metric('f1 score', f1_score(y_test, y_pred))
        
        # Registro do Dataset
        mlflow.log_input(mlflow_dataset, context='raw-data')
        
        # Registro de tags para fins de documentação
        mlflow.set_tag('Training info', f'Basic {estimator_name} for heart disease data')
        mlflow.set_tag('Model origin', 'This estimator is a pipeline')
        
        signature = infer_signature(X_train, pipeline.predict(X_test))
        
        # Registro do modelo em si
        # A variável model_info contém informações retornadas após o tracking do modelo
        model_info = mlflow.sklearn.log_model(
            name=estimator_name,
            sk_model=pipeline,
            signature=signature, 
            input_example=X_test.iloc[[0]],
            registered_model_name=estimator_name
        )

        return model_info
