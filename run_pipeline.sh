#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-./bin/python}"

"$PYTHON" model-training/ml_pipeline_graph.py "$@"
