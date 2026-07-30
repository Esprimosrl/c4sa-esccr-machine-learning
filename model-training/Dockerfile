FROM python:3.12-slim AS base

ENV PYTHON=python \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY model-training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 app \
    && install -d -o app -g app /artifacts

COPY run_pipeline.sh quantumleap.py orion.py lag_features.py inference_data.py ./
COPY model-training/ model-training/
COPY dataset/distribuzione_giornaliera.py dataset/
COPY data_csv_parquet/calendario_raccolta.csv data_csv_parquet/

USER app

FROM base AS api

EXPOSE 8000
HEALTHCHECK CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]
CMD ["uvicorn", "training_api:app", "--app-dir", "model-training", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

FROM base AS batch

ENTRYPOINT ["./run_pipeline.sh"]
