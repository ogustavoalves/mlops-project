from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import mlflow
import os
from load_model import get_model 

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
app = FastAPI()

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
    try:
        decision_tree_model = get_model()
    except Exception as e:
        return {"error": "Model unavailable.", "detail": str(e)}
    
    # Construir Dataframe de features
    # Columnas
    columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    # Valores
    features = pd.DataFrame([[
            data.age, data.sex, data.cp, data.trestbps,
            data.chol, data.fbs, data.restecg, data.thalach,
            data.exang, data.oldpeak, data.slope, data.ca, data.thal
        ]], columns=columns)

    y_pred = decision_tree_model.predict(features)
    
    return {'prediction': int(y_pred[0])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
