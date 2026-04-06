import mlflow.sklearn
import mlflow
import os

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

decision_tree_model = None

def get_model():
    global decision_tree_model
    
    if decision_tree_model is None:
        decision_tree_model = mlflow.sklearn.load_model(
            'models:/DecisionTreeClassifier/Production'
        )
        print('Modelo carregado sob demanda')
        
    return decision_tree_model