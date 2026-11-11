from fastapi.testclient import TestClient
from server import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"healthy": "true"}

def test_predict_validation_error():
    payload = {
        "age": "invalido", "sex": 1, "cp":0, "trestbps": 140,
        "chol": 211, "fbs": 1, "restecg": 1, "thalach": 165,
        "exang": 1, "oldpeak": 0.6, "slope": 2, "ca": 0, "thal": 2
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 