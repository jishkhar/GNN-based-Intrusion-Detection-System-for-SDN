#!/usr/bin/env bash
# End-to-end Phase 2 run: IDS API on the host, controller + Mininet + attacks in
# the lab container, then mitigation analysis and labelling of the recorded flows.
#
#   bash scripts/run_phase2_demo.sh                       # every attack once, 45 s each
#   SCENARIO=ddos DURATION=60 REPEAT=5 bash scripts/run_phase2_demo.sh
#   KEEP_IDS=1 bash scripts/run_phase2_demo.sh            # leave the dashboard running afterwards
#   COLLECT=1 REPEAT=5 bash scripts/run_phase2_demo.sh    # record labelled data, no mitigation
#   USE_RUNNING_IDS=1 bash scripts/run_phase2_demo.sh     # reuse the IDS already on :$PORT (scripts/run_all.sh);
#                                                         # it must have been started with this session's
#                                                         # IDS_API_KEY, IDS_LOG_DIR and IDS_RECORD_FLOWS
#
# Needs: Docker (user in the docker group), the gnn-ids-lab image
# (docker build -t gnn-ids-lab -f docker/Dockerfile.sdn-lab docker/) and the
# exported model (models/gat_ids.pt, from scripts/run_phase2_training.sh).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-.venv/bin/python}
SCENARIO=${SCENARIO:-all}
DURATION=${DURATION:-45}
WARMUP=${WARMUP:-25}
COOLDOWN=${COOLDOWN:-25}
REPEAT=${REPEAT:-1}
DATAPATH=${DATAPATH:-kernel}
PORT=${PORT:-3000}
SESSION=${SESSION:-data/mininet/$(date +%Y%m%d_%H%M%S)}
export IDS_API_KEY=${IDS_API_KEY:-lab-$(date +%s)}

mkdir -p "$SESSION/runs"
export IDS_LOG_DIR="$PWD/$SESSION/logs"
export IDS_RECORD_FLOWS="$PWD/$SESSION/flows.csv"
export PYTHONWARNINGS=ignore
EXTRA_ARGS=""
if [[ "${COLLECT:-0}" == "1" ]]; then
  export IDS_MITIGATION=0   # blocks would change the traffic being recorded
  EXTRA_ARGS="--vary-rate"
fi

IDS_PID=""
if [[ "${USE_RUNNING_IDS:-0}" == "1" ]]; then
  echo "[1/5] Using the IDS API already running on :$PORT (session $SESSION)"
else
  echo "[1/5] Starting IDS API on :$PORT (session $SESSION)"
  (cd web && exec "../$PY" -m uvicorn app:app --host 127.0.0.1 --port "$PORT" > "../$SESSION/ids_api.log" 2>&1) &
  IDS_PID=$!
fi
cleanup() { if [[ -n "$IDS_PID" && "${KEEP_IDS:-0}" != "1" ]]; then kill "$IDS_PID" 2>/dev/null || true; fi; }
trap cleanup EXIT
for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/api/health" > /dev/null && break
  sleep 1
done
curl -s "http://127.0.0.1:$PORT/api/health" | grep -q '"live_ids":true' || {
  echo "IDS failed to start"; [[ -f "$SESSION/ids_api.log" ]] && tail "$SESSION/ids_api.log"; exit 1; }

echo "[2/5] Lab: controller + Mininet + scenario '$SCENARIO' ($REPEAT x ${DURATION}s)"
docker run --rm --privileged --network host -v "$PWD":/work \
  -e IDS_API_KEY -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" gnn-ids-lab bash -c "
    python3 controller/run_controller.py --ids-url http://127.0.0.1:$PORT/api/flows > $SESSION/controller.log 2>&1 &
    sleep 4
    python3 scripts/inject_attack.py --scenario $SCENARIO --duration $DURATION --warmup $WARMUP \
      --cooldown $COOLDOWN --repeat $REPEAT --datapath $DATAPATH --out-dir $SESSION/runs $EXTRA_ARGS
    status=\$?
    chown -R \$HOST_UID:\$HOST_GID $SESSION
    exit \$status"

echo "[3/5] Mitigation analysis"
$PY scripts/analyze_mitigation.py --runs "$SESSION/runs/*.json" --log "$SESSION/logs/mitigation_log.jsonl" \
  --out "$SESSION/mitigation_report.json"

echo "[4/5] Labelling recorded flows -> graphs"
$PY scripts/label_mininet_flows.py --flows "$SESSION/flows.csv" --runs "$SESSION/runs/*.json" \
  --out "$SESSION/mininet_v2.pt"

echo "[5/5] Transfer check: InSDN-trained model on this Mininet traffic"
$PY -m models.evaluate_v2 --checkpoint-path models/checkpoints/gnn_v2_gat.pt \
  --graph-path "$SESSION/mininet_v2.pt" --split all --out "$SESSION/transfer_eval.json"

echo "Done. Session files in $SESSION/"
if [[ -n "$IDS_PID" && "${KEEP_IDS:-0}" == "1" ]]; then
  echo "IDS still running: http://127.0.0.1:$PORT/live (API key: $IDS_API_KEY, pid $IDS_PID)"
  trap - EXIT
fi
