# MLOps Project

This project implements a simple MLOps pipeline using Docker, MLflow and a API (FastAPI) to serve the inference of the model.
## Running the project

No diretório raiz — onde está localizado o arquivo `docker-compose.yml` — execute:
In the same directory level to the file `docker-compose.yml`, run the command:

```bash
docker compose up --build
```

### Access to the MLflow UI

Once the containers are running, open your browser and access:

`http://localhost:5000`

### Testing Inference 

Inference method will be available at:

`POST http://localhost:8000/predict`

*JSON Payload example:* 
```json
{
  "age": 54,
  "sex": 1,
  "cp": 0,
  "trestbps": 120,
  "chol": 188,
  "fbs": 0,
  "restecg": 1,
  "thalach": 113,
  "exang": 0,
  "oldpeak": 1.4,
  "slope": 1,
  "ca": 1,
  "thal": 3
}
```

The model's answer will be `True` (1) or `False` (0), indicating whether the patient has heart disease, based on the input data.


