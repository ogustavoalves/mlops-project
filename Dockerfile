# Base image
FROM python:3.10.12-slim

#Create the /app directory inside of the container
WORKDIR /app

#Copy the requirements
COPY requirements.txt .

#Install the requirements
RUN pip install -r requirements.txt

#Copy the entire content of the current folder
COPY . .

EXPOSE 5000

#Runtime command -- runs mlflow server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "./mlruns", "--default-artifact-root", "file:./mlruns"]
