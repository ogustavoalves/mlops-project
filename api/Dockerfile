FROM python:3.10.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["/bin/sh", "-c", "test -f server.py || { echo '❌ server.py not found!'; exit 1; }; uvicorn server:app --host 0.0.0.0 --port 8000"]