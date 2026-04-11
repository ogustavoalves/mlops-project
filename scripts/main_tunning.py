import os
import mlflow
import traceback
import pandas as pd
import pickle
from datetime import date
from sklearn.model_selection import train_test_split
from mlflow_manager import train_and_log_model
from promote_model import promote_champion
from config import PIPELINES, PIPELINES_FOR_TUNNING, EXPERIMENT_NAME, CHAMPION_MODEL
from model_tuner import generic_grid_search



print('Tunning and training script started.')

try:
    # Leitura dos dados
    dataset_path = os.path.abspath('data/raw/heart.csv') # Removido '../' para funcionar dentro do container
    dataset = pd.read_csv(dataset_path)
except Exception as e:
    print(f'Error loading data: {e}')
else:
    print('Data successfuly loaded.')
    print()
    
# Definir o tracking ui
mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
mlflow.set_experiment(EXPERIMENT_NAME)

X = dataset.iloc[:, 0:13]
y = dataset.iloc[:, 13]

X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size=0.2, random_state=0) 

print('Tunning phase started.')
print()
best_params = {}
# Treinamento e registro MLflow
for model_name, (pipeline, params) in PIPELINES_FOR_TUNNING.items():
    print(f'Tunning model: {model_name}')
    
    try:
        best_params, estimator_name = generic_grid_search(
            pipeline=pipeline,
            params=params,
            X_matrix=X_train,
            y_matrix=y_train
        )
        
        best_params[f'{model_name}_best_params'] = (pipeline, best_params)
        
    except Exception as e:
        print(f'Error while tunning: {model_name}')
        traceback.print_exc()
    else:
        print(f'Success!')
        print(f'Best params: {best_params}.')
        print()
        # 
        current_date = date.today()
        with open(f'../data/raw/best_params_{current_date}.pkl', 'wb') as f:
            pickle.dump(best_params, f)

print('Training phase started.')
print()
# Treinamento e registro MLflow
for model_name, (pipeline, param) in PIPELINES.items():
    print(f'Training: {model_name}')
    
    try:
        model_info = train_and_log_model(
            pipeline,
            param,
            X_train,
            X_test,
            y_train,
            y_test,
            dataset,
            dataset_path
        )
    except Exception as e:
        print(f'Error: {e}')
        traceback.print_exc()
    else:
        print(f'Successfully trained and registered model {model_name}.')
        print(f'Artifact path: {model_info.artifact_path}.')
        print()

promote_champion(model_name=CHAMPION_MODEL)