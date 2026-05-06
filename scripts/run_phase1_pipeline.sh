#!/usr/bin/env bash
set -euo pipefail

echo "[1/6] Cleaning CICIDS2017 raw CSV files"
python preprocessing/clean_data.py \
  --input-glob "data/cicids2017/raw/*.csv" \
  --output-dir "data/cicids2017/cleaned" \
  --label-column "Label"

echo "[2/6] Cleaning InSDN raw CSV files"
python preprocessing/clean_data.py \
  --input-glob "data/insdn/raw/*.csv" \
  --output-dir "data/insdn/cleaned" \
  --label-column "Label"

echo "[3/6] Running baseline models (RF/XGBoost) on CICIDS cleaned data"
python baselines/train_baselines.py \
  --input-glob "data/cicids2017/cleaned/*.csv" \
  --label-col "Label" \
  --metrics-out "results/baseline_metrics.json" \
  --cm-out "results/baseline_confusion_matrix.png"

echo "[4/6] Building graph snapshots"
python preprocessing/graph_builder.py \
  --input-glob "data/cicids2017/cleaned/*.csv" \
  --output-path "data/graphs/cicids_graphs.pt" \
  --label-col "Label" \
  --window-seconds 5 \
  --step-seconds 1

echo "[5/6] Training GNN"
python models/train_gnn.py \
  --graph-path "data/graphs/cicids_graphs.pt" \
  --checkpoint-path "models/checkpoints/best_gat.pt" \
  --metrics-path "results/gnn_metrics.json" \
  --epochs 40

echo "[6/6] Evaluating GNN"
python models/evaluate.py \
  --graph-path "data/graphs/cicids_graphs.pt" \
  --checkpoint-path "models/checkpoints/best_gat.pt" \
  --report-out "results/gnn_classification_report.json" \
  --cm-out "results/gnn_confusion_matrix.png"

echo "Phase 1 pipeline complete. Check results/ for outputs."
