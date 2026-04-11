import os
import pandas as pd
import mlflow
import traceback
from sklearn.model_selection import train_test_split
from mlflow_manager import train_and_log_model
from promote_model import promote_champion
from config import PIPELINES, EXPERIMENT_NAME, CHAMPION_MODEL

print('Training script started.')

try:
    # Read dataset
    dataset_path = os.path.abspath('data/raw/heart.csv') 
    dataset = pd.read_csv(dataset_path)
except Exception as e:
    print(f'Error loading data: {e}')
else:
    print('Data successfuly loaded.')
    print()

# Setting up MLflow's tracking ui
mlflow.set_tracking_uri(uri='http://localhost:5000')
# Setting up experiment's name
response = mlflow.set_experiment('Heart disease pipeline experiment')

X = dataset.iloc[:, 0:13]
y = dataset.iloc[:, 13]

X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size=0.2, random_state=0) 

print('Training phase started.')
print()

for model_name, (pipeline, param) in PIPELINES.items():
    print(f'Model: {model_name}')
    try:
        model_tacking = train_and_log_model(
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
        print(f'Successfully trained and registered model: {model_name}')
        print(f'Artifact path: {model_tacking.artifact_path}')
        print()

promote_champion(model_name=CHAMPION_MODEL)