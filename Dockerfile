# Base image
FROM python:3.10.12-slim

#Create the /app directory inside of the container
WORKDIR /app

#Copy the requirements
COPY requirements.txt .

#Install the requirements
RUN pip install -r requirements.txt

RUN mkdir -p /app/mlruns && touch /app/mlflow.db && chmod -R 777 /app

EXPOSE 5000

#Runtime command -- runs mlflow server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:////app/mlflow.db", "--artifacts-destination", "/app/mlruns", "--serve-artifacts"]
