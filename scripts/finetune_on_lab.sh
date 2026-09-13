#!/usr/bin/env bash
# Fine-tune the GNN on labelled Mininet lab traffic and install it as the live model.
#
#   bash scripts/finetune_on_lab.sh data/mininet/collect1 data/mininet/collect_probe
#
# Each argument is a lab session recorded with COLLECT=1 (scripts/run_phase2_demo.sh).
# Sessions are labelled if needed, merged (runs stay split whole), the model is trained on
# InSDN + lab train runs, evaluated on the lab and InSDN test sets, and exported to
# models/gat_ids.pt (the file the live IDS loads).
set -euo pipefail
cd "$(dirname "$0")/.."
[[ $# -ge 1 ]] || { echo "usage: $0 SESSION_DIR [SESSION_DIR ...]"; exit 1; }

PY=${PYTHON:-.venv/bin/python}
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
MERGED=${MERGED:-data/mininet/mininet_all_v2.pt}
CKPT=${CKPT:-models/checkpoints/gnn_v2_gat_ft.pt}
MODEL_OUT=${MODEL_OUT:-models/gat_ids.pt}
METRICS=${METRICS:-results/phase2/gnn_v2_gat_ft.json}
EPOCHS=${EPOCHS:-60}

echo "[1/4] Labelling sessions"
for s in "$@"; do
  if [[ ! -f "$s/mininet_v2.pt" ]]; then
    $PY scripts/label_mininet_flows.py --flows "$s/flows.csv" --runs "$s/runs/*.json" --out "$s/mininet_v2.pt"
  fi
done
$PY - "$MERGED" "$@" <<'PYEOF'
import sys, torch
out_path, sessions = sys.argv[1], sys.argv[2:]
parts = [torch.load(f"{s}/mininet_v2.pt", weights_only=False) for s in sessions]
merged = {k: sum((p[k] for p in parts), []) for k in ("train", "val", "test")}
merged["meta"] = {**parts[0]["meta"], "sources": sessions}
torch.save(merged, out_path)
print({k: len(merged[k]) for k in ("train", "val", "test")}, "->", out_path)
PYEOF

echo "[2/4] Fine-tuning on InSDN + lab data"
$PY -m models.train_gnn_v2 --config configs/phase2.yaml --extra-graph-paths "$MERGED" --extra-oversample 2 \
  --epochs "$EPOCHS" --checkpoint-path "$CKPT" --metrics-path "$METRICS" --scores-out "" --curves-path ""

echo "[3/4] Evaluating on held-out lab runs and on InSDN"
$PY -m models.evaluate_v2 --checkpoint-path "$CKPT" --graph-path "$MERGED" --split test \
  --out results/phase2/mininet_detection_after_finetune.json
$PY -m models.evaluate_v2 --checkpoint-path "$CKPT" --graph-path data/graphs/insdn_v2.pt --split test \
  --out results/phase2/finetuned_on_insdn_test.json

echo "[4/4] Exporting live model -> $MODEL_OUT"
$PY -m models.export --checkpoint-path "$CKPT" --output-path "$MODEL_OUT" --graph-path "$MERGED"
