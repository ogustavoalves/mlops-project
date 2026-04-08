import os
import mlflow
import pandas as pd
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
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
    debug: Optional[bool] = False

@app.get('/health', status_code=200)
async def health_check():
    return {'healthy': 'true'}

@app.get('/')
def welcome_message():
    return {'message': "FastAPI server is up."}

@app.post('/predict')
def predict (data: InputData):
    try:
        model = get_model()
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

    if data.debug:
        y_pred_proba = model.predict_proba(features)
        res = {int(cls): float(prob) for cls, prob in zip(model.classes_, y_pred_proba[0])}
        
        return {'prediction': res}
    
    y_pred = model.predict(features)
    return {'prediction': int(y_pred[0])}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
