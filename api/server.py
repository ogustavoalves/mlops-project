from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import numpy as np
import mlflow
import os

# mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
app = FastAPI()

try: 
    decision_tree_model = mlflow.sklearn.load_model('models:/Decision-tree-classifier/1')
    print('Modelo carregado com sucesso')
except Exception as e:
    print('Erro ao carregar o modelo:', e)
    decision_tree_model = None

class InputData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.get('/')
def welcome_message():
    return {'message': "API is working!"}

@app.post('/predict')
def predict (data: InputData):
    features = np.array([[data.age, data.sex, data.cp, data.trestbps, data.chol, data.fbs, data.restecg, data.thalach, data.exang, data.oldpeak, data.slope, data.ca, data.thal]])

    y_pred = decision_tree_model.predict(features)
    
    return {'prediction': int(y_pred[0])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
