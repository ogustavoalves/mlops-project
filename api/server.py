import os
import mlflow
import pandas as pd
from typing import Optional
from fastapi import FastAPI, Response, Request
from pydantic import BaseModel
from load_model import get_model
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
app = FastAPI()

# Prometheus client 
# counter of total predictions
PREDICTIONS_TOTAL = Counter('api_predictions', "Total number of predictions made")
INFERENCE_TIME = Histogram('api_inference_duration_seconds', "Time of each inference made")
API_RESPONSES_TOTAL = Counter(
        'api_responses_total',
        "Total de respostas da API por código de estado HTTP",
        ['status_code']
)

@app.middleware("http")
async def monitor_http_status(request: Request, call_next):
    response = await call_next(request)
    
    API_RESPONSES_TOTAL.labels(status_code=str(response.status_code)).inc()
    
    return response
        
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
    # Columns
    columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    # Values  ]
    features = pd.DataFrame([[
            data.age, data.sex, data.cp, data.trestbps,
            data.chol, data.fbs, data.restecg, data.thalach,
            data.exang, data.oldpeak, data.slope, data.ca, data.thal
        ]], columns=columns)

    # Increments the counter when a prediction is made
    PREDICTIONS_TOTAL.inc()
    with INFERENCE_TIME.time():
        if data.debug:
            y_pred_proba = model.predict_proba(features)
            res = {int(cls): float(prob) for cls, prob in zip(model.classes_, y_pred_proba[0])}
            
            return {'prediction': res}
        
        y_pred = model.predict(features)
        return {'prediction': int(y_pred[0])}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
