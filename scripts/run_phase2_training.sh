#!/usr/bin/env bash
# Phase 2 offline pipeline: graphs -> baselines -> GNN -> ablations -> export.
# Usage: bash scripts/run_phase2_training.sh [--skip-ablations]
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=configs/phase2.yaml
PY=${PYTHON:-python}
export PYTHONUNBUFFERED=1
OUT=results/phase2
mkdir -p "$OUT" models/checkpoints

echo "[1/5] Building InSDN topology graphs"
$PY -m preprocessing.graph_builder_v2 --config "$CONFIG"

echo "[2/5] Baselines (RF / XGBoost) on the same features and split"
$PY -m baselines.train_baselines_v2 --config "$CONFIG"

echo "[3/5] Main model: GAT with edge features"
$PY -m models.train_gnn_v2 --config "$CONFIG"

if [[ "${1:-}" != "--skip-ablations" ]]; then
  echo "[4/5] Ablations"
  for conv in gcn sage; do
    $PY -m models.train_gnn_v2 --config "$CONFIG" --conv-type "$conv" \
      --checkpoint-path "models/checkpoints/gnn_v2_${conv}.pt" \
      --metrics-path "$OUT/ablation_${conv}.json" --scores-out "" --curves-path ""
  done
  $PY -m models.train_gnn_v2 --config "$CONFIG" --no-edge-features \
    --checkpoint-path models/checkpoints/gnn_v2_gat_noedge.pt \
    --metrics-path "$OUT/ablation_gat_noedge.json" --scores-out "" --curves-path ""

  for size in 50 200; do
    stride=$((size / 4))
    $PY -m preprocessing.graph_builder_v2 --config "$CONFIG" \
      --window-size "$size" --window-stride "$stride" \
      --output-path "data/graphs/insdn_v2_w${size}.pt"
    $PY -m models.train_gnn_v2 --config "$CONFIG" --graph-path "data/graphs/insdn_v2_w${size}.pt" \
      --checkpoint-path "models/checkpoints/gnn_v2_gat_w${size}.pt" \
      --metrics-path "$OUT/ablation_window_${size}.json" --scores-out "" --curves-path ""
  done
else
  echo "[4/5] Ablations skipped"
fi

echo "[5/5] Export TorchScript model"
$PY -m models.export --config "$CONFIG"
# Never overwrite a fine-tuned live model; only provide one if none exists yet.
if [[ ! -f models/gat_ids.pt ]]; then
  cp models/gat_ids_insdn_only.pt models/gat_ids.pt
  cp models/gat_ids_insdn_only.json models/gat_ids.json
  echo "models/gat_ids.pt <- InSDN-only model (run scripts/finetune_on_lab.sh before live use)"
else
  echo "models/gat_ids.pt kept (live model); InSDN-only export is models/gat_ids_insdn_only.pt"
fi

echo "Phase 2 offline pipeline complete. See $OUT/"
