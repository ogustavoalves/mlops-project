# MLOps Project

MLOps pipeline for heart disease classification using the [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) from Kaggle. The project trains, registers, and serves multiple classification models via REST API, orchestrating everything with Docker Compose.

## Architecture

```
docker compose up
       │
       ▼
   [MLflow] ──────────────────────────────┐
       │  (service_healthy)               │
       ▼                                  │
[training-script]                         │ tracking + artifacts
  - Trains 7 models                       │
  - Registers in MLflow                   │
  - Promotes champion                     │
       │  (service_completed_successfully)│
       ▼                                  │
   [FastAPI] ─────────────────────────────┘
  - Loads model from MLflow
  - Serves predictions at /predict
```

## Trained models

- Decision Tree
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors
- Logistic Regression
- SVC
- MLP Classifier

Each model has its own preprocessor suited to the algorithm type (OrdinalEncoder for tree-based models, OneHotEncoder + StandardScaler for the others).

## Project structure

```
mlops_project/
├── api/
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
├── scripts/
│   ├── Dockerfile
│   ├── main_train.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── mlflow_manager.py
│   ├── promote_model.py
│   └── requirements.txt
├── data/
│   └── raw/
│       └── heart.csv
├── Dockerfile          # MLflow server
├── docker-compose.yml
└── requirements.txt
```

## How to run

In the root directory, run:

```bash
docker compose up --build
```

The startup order is managed automatically:
1. MLflow starts and waits to become healthy
2. Training script trains and registers the models
3. FastAPI starts with the models available

## Accessing the services

| Service | URL |
|---|---|
| MLflow UI | http://localhost:5000 |
| Inference API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

## Testing inference

Inference method will be available at:

`POST http://localhost:8000/predict`

*JSON input example:* 
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
>>>>>>> 2716189 (Fix formatting issues in README.md)
```

**Expected response:**
```json
{"prediction": 1}
```

To get the probabilities per class, add `"debug": true` to the payload:
```json
{"prediction": {"0": 0.23, "1": 0.77}}
```

## Running MLflow individually

On the first run, create the database file before starting the container:

```bash
touch mlflow.db
docker run -p 5000:5000 \
  -v "$(pwd)/mlruns:/app/mlruns" \
  -v "$(pwd)/mlflow.db:/app/mlflow.db" \
  mlops-project/mlflow:1.2
```

> On subsequent runs, `touch mlflow.db` can be omitted.

## Dataset features

| Feature | Description |
|---|---|
| age | Patient age |
| sex | Sex (1 = male, 0 = female) |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (1 = true) |
| restecg | Resting electrocardiographic results (0–2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina (1 = yes) |
| oldpeak | ST depression induced by exercise |
| slope | Slope of the peak exercise ST segment (0–2) |
| ca | Number of major vessels colored (0–3) |
| thal | Thalassemia (0–3) |

**Target:** `1` = presence of heart disease, `0` = absence
