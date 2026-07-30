FROM python:3.12-slim

ENV MODEL_DIR=/models \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY onnx_inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 app
COPY onnx_inference/app.py .
COPY quantumleap.py .
COPY inference_data.py orion.py lag_features.py .
COPY model-training/zone_fraction_benchmark.py .
COPY data_csv_parquet/calendario_raccolta.csv .

USER app
EXPOSE 8000
HEALTHCHECK CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
