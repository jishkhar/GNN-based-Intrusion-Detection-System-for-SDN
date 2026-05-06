#!/usr/bin/env bash
set -euo pipefail

echo "[1/6] Cleaning CICIDS2017 raw CSV files"
python -m preprocessing.clean_data \
  --input-glob "data/cicids2017/raw/*.csv" \
  --output-dir "data/cicids2017/cleaned" \
  --label-column "Label"

echo "[2/6] Cleaning InSDN raw CSV files"
python -m preprocessing.clean_data \
  --input-glob "data/insdn/raw/*.csv" \
  --output-dir "data/insdn/cleaned" \
  --label-column "Label"

echo "[3/6] Running baseline models (RF/XGBoost) on CICIDS cleaned data"
python -m baselines.train_baselines \
  --input-glob "data/cicids2017/cleaned/*.csv" \
  --label-col "Label" \
  --metrics-out "results/baseline_metrics.json" \
  --cm-out "results/baseline_confusion_matrix.png"

echo "[4/6] Building graph snapshots"
python -m preprocessing.graph_builder \
  --input-glob "data/cicids2017/cleaned/*.csv" \
  --output-path "data/graphs/cicids_graphs.pt" \
  --label-col "Label" \
  --window-seconds 5 \
  --step-seconds 1 \
  --max-graphs-per-file 5000

echo "[5/6] Training GNN"
python -m models.train_gnn \
  --graph-path "data/graphs/cicids_graphs.pt" \
  --checkpoint-path "models/checkpoints/best_gat.pt" \
  --metrics-path "results/gnn_metrics.json" \
  --batch-size 128 \
  --epochs 40

echo "[6/6] Evaluating GNN"
python -m models.evaluate \
  --graph-path "data/graphs/cicids_graphs.pt" \
  --checkpoint-path "models/checkpoints/best_gat.pt" \
  --report-out "results/gnn_classification_report.json" \
  --cm-out "results/gnn_confusion_matrix.png"

echo "Phase 1 pipeline complete. Check results/ for outputs."
