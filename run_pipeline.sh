#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-./bin/python}"
SEED="${SEED:-42}"
ANNO="${ANNO:-2024}"

"$PYTHON" model-training/classification/train_kmeans.py \
  --input data_csv_parquet/andria_distribuzione_annuale_zone_inferite_e_zone_costruzione.csv \
  --n-clusters 6 \
  --output data_csv_parquet/kmeans_assignments_annuo_peso_coord.csv \
  --output-format csv \
  --model-output models/kmeans_model_annuo_peso_coord.joblib \
  --coord-weight 8 \
  --annual-kg-weight 0.5 \
  --other-numeric-weight 1 \
  --categorical-weight 1

"$PYTHON" dataset/distribuzione_giornaliera.py \
  --annuo data_csv_parquet/kmeans_assignments_annuo_peso_coord.csv \
  --anno "$ANNO" \
  --calendario data_csv_parquet/calendario_raccolta.csv \
  --id-col "id_civico" \
  --zona-col "zona_raccolta" \
  --seed "$SEED" \
  --out data_csv_parquet/conferimenti_giornalieri_classificati.csv \
  --out-produzione data_csv_parquet/produzione_giornaliera_classificata.csv \
  --missing-calendar-mode daily

"$PYTHON" model-training/regression/train_grouped_sgd_time_series_regressor.py \
  --input data_csv_parquet/conferimenti_giornalieri_classificati.csv \
  --split-ratios 80,10,10 \
  --epochs 3 \
  --batch-size 4096 \
  --predictions-output predictions/sgd_test_predictions_grouped.parquet \
  --progress
