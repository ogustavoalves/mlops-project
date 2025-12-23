FROM python:3.10.12-slim

# 
ENV PYTHONDONTWRITEBYTECODE=1 

WORKDIR /app-script

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/*.py ./
COPY data/raw/heart.csv ./data/raw/


CMD ["python3", "main.py"]

