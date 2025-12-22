# MLOps Project

Este projeto implementa um pipeline simples de MLOps utilizando Docker, MLflow e uma API de inferência para servir o modelo.

## Como executar o projeto

No diretório raiz — onde está localizado o arquivo `docker-compose.yml` — execute:

*Iniciar serviços*
```bash
docker compose up --build
```
*Ajuste de hiperparâmetros, treinamentos e registros no MLflow*
Em seguida, rode:
```bash
python3 main.py
```

### Acessando MLflow UI
Com os containers ativos, abra o navegador e acesse:
`http://localhost:5000`

### Testando a inferência do modelo
A API de inferência estará disponível no endpoint:
`POST http://localhost:8000/predict`

**Exemplo de payload JSON
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
A resposta retornará a predição do modelo para o caso enviado.


