FROM python:3.12-slim

ENV PYTHON=python \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY model-training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 app \
    && install -d -o app -g app /artifacts

COPY run_pipeline.sh quantumleap.py orion.py ./
COPY model-training/ml_pipeline_graph.py model-training/prepare_quantumleap.py model-training/publish_digital_twins.py model-training/
COPY model-training/classification/train_kmeans.py model-training/classification/kmeans_feature_tables.py model-training/classification/
COPY model-training/regression/train_grouped_sgd_time_series_regressor.py model-training/regression/modular_hist_gradient_boosting_regressor.py model-training/regression/
COPY dataset/distribuzione_giornaliera.py dataset/
COPY data_csv_parquet/calendario_raccolta.csv data_csv_parquet/

USER app

ENTRYPOINT ["./run_pipeline.sh"]
