# MLOps Project

Pipeline de MLOps para classificação de doenças cardíacas utilizando o [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) do Kaggle. O projeto treina, registra e serve múltiplos modelos de classificação via API REST, orquestrando tudo com Docker Compose.

## Arquitetura

```
docker compose up
       │
       ▼
   [MLflow] ──────────────────────────────┐
       │  (service_healthy)               │
       ▼                                  │
[training-script]                         │ tracking + artifacts
  - Treina 7 modelos                      │
  - Registra no MLflow                    │
  - Promove campeão                       │
       │  (service_completed_successfully)│
       ▼                                  │
   [FastAPI] ─────────────────────────────┘
  - Carrega modelo do MLflow
  - Serve predições em /predict
```

## Modelos treinados

- Decision Tree
- Random Forest
- Gradient Boosting
- K-Nearest Neighbors
- Logistic Regression
- SVC
- MLP Classifier

Cada modelo possui seu próprio preprocessador adequado ao tipo de algoritmo (OrdinalEncoder para modelos baseados em árvore, OneHotEncoder + StandardScaler para os demais).

## Estrutura do projeto

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

## Como executar

No diretório raiz, execute:

```bash
docker compose up --build
```

A ordem de inicialização é gerenciada automaticamente:
1. MLflow sobe e aguarda ficar saudável
2. Training script treina e registra os modelos
3. FastAPI sobe com os modelos disponíveis

## Acessando os serviços

| Serviço | URL |
|---|---|
| MLflow UI | http://localhost:5000 |
| API de inferência | http://localhost:8000 |
| Docs da API | http://localhost:8000/docs |

## Testando a inferência

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63,
    "sex": 1,
    "cp": 0,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
  }'
```

**Resposta esperada:**
```json
{"prediction": 1}
```

Para obter as probabilidades por classe, adicione `"debug": true` ao payload:
```json
{"prediction": {"0": 0.23, "1": 0.77}}
```

## Subir o MLflow individualmente

Na primeira execução, crie o arquivo de banco antes de subir o container:

```bash
touch mlflow.db
docker run -p 5000:5000 \
  -v "$(pwd)/mlruns:/app/mlruns" \
  -v "$(pwd)/mlflow.db:/app/mlflow.db" \
  mlops-project/mlflow:1.2
```

> Nas execuções seguintes o `touch mlflow.db` pode ser omitido.

## Features do dataset

| Feature | Descrição |
|---|---|
| age | Idade do paciente |
| sex | Sexo (1 = masculino, 0 = feminino) |
| cp | Tipo de dor no peito (0–3) |
| trestbps | Pressão arterial em repouso (mm Hg) |
| chol | Colesterol sérico (mg/dl) |
| fbs | Glicemia em jejum > 120 mg/dl (1 = verdadeiro) |
| restecg | Resultado do eletrocardiograma em repouso (0–2) |
| thalach | Frequência cardíaca máxima atingida |
| exang | Angina induzida por exercício (1 = sim) |
| oldpeak | Depressão do ST induzida por exercício |
| slope | Inclinação do segmento ST de pico (0–2) |
| ca | Número de vasos principais coloridos (0–3) |
| thal | Talassemia (0–3) |

**Target:** `1` = presença de doença cardíaca, `0` = ausência