#!/usr/bin/env bash
# Master script: runs every phase of the project in order and shows live progress on the dashboard.
#
#   bash scripts/run_all.sh                    # everything: Phase 1, Phase 2 (with ablations), tests,
#                                              # replay, latency, live Mininet demo, report
#   bash scripts/run_all.sh --quick            # skip the Phase 2 ablations (~15 min of training instead of ~70)
#   bash scripts/run_all.sh --skip-training    # reuse the trained models: tests, replay, live demo, report
#
# While it runs, open http://127.0.0.1:3000/pipeline (stage progress, live log, headline results).
# During the replay and live-demo stages, http://127.0.0.1:3000/live shows detections and blocks.
# When everything is done the dashboard stays up until Ctrl+C.
#
# Stages:
#    1 preflight     Python packages, datasets, Docker, free port
#    2 dashboard     start the web app (offline results + live IDS + pipeline progress)
#    3 phase1        Phase 1 on CICIDS2017: clean -> baselines -> graphs -> GAT -> evaluation
#    4 clean_insdn   Phase 2 cleaning of InSDN
#    5 phase2_train  InSDN graphs -> baselines -> multi-task GNN -> ablations -> TorchScript export
#    6 tests         pytest suite
#    7 replay        held-out InSDN flows replayed through the running IDS (visible on /live)
#    8 latency       inference latency benchmark
#    9 live_demo     Docker lab: os-ken controller + Mininet + attacks, blocked live (visible on /live)
#   10 report        results/phase2/final_results.md
#
# Options:
#   --quick           skip the Phase 2 ablations
#   --skip-phase1     skip the Phase 1 pipeline
#   --skip-training   skip cleaning and training in both phases (needs models/gat_ids.pt)
#   --skip-live       skip the Docker/Mininet live demo
#   --no-serve        stop the dashboard when finished instead of keeping it up
#
# Environment: PORT (3000), PYTHON (.venv/bin/python); for the live demo SCENARIO (all), DURATION (45),
# REPEAT (1), WARMUP (25), COOLDOWN (25), DATAPATH (kernel; user if the openvswitch module is missing).
set -uo pipefail  # no -e: each stage's failure is handled explicitly
cd "$(dirname "$0")/.."
ROOT=$PWD

QUICK=0 SKIP_PHASE1=0 SKIP_TRAINING=0 SKIP_LIVE=0 SERVE=1
for arg in "$@"; do
  case $arg in
    --quick) QUICK=1 ;;
    --skip-phase1) SKIP_PHASE1=1 ;;
    --skip-training) SKIP_TRAINING=1 ;;
    --skip-live) SKIP_LIVE=1 ;;
    --no-serve) SERVE=0 ;;
    -h|--help) sed -n '2,/^set -uo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $arg (see --help)"; exit 2 ;;
  esac
done

PY=${PYTHON:-.venv/bin/python}
[[ $PY == */* && $PY != /* ]] && PY=$ROOT/$PY
PORT=${PORT:-3000}
DASH_URL=http://127.0.0.1:$PORT
RUN_ID=$(date +%Y%m%d_%H%M%S)
LOG_DIR=logs/pipeline/$RUN_ID
STATUS=logs/pipeline/status.json
SESSION=data/mininet/run_all_$RUN_ID
export PATH="$ROOT/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
export IDS_API_KEY=${IDS_API_KEY:-$(head -c 12 /dev/urandom | od -An -tx1 | tr -d ' \n')}
DASH_PID=""

# ---------------------------------------------------------------- checks before anything starts
command -v "$PY" > /dev/null || { echo "Python not found: $PY (create the venv first, see Execution.md)"; exit 1; }
if (exec 3<> "/dev/tcp/127.0.0.1/$PORT") 2> /dev/null; then
  echo "Port $PORT is already in use (an earlier dashboard or demo?)."
  command -v ss > /dev/null && ss -ltnp 2> /dev/null | grep ":$PORT " | sed 's/^/  /'
  echo "Stop that process, or run on another port: PORT=3001 bash scripts/run_all.sh"
  exit 1
fi

has_files() { compgen -G "$1" > /dev/null; }
DOCKER_OK=0
docker info > /dev/null 2>&1 && DOCKER_OK=1
DATAPATH=${DATAPATH:-$([[ -d /sys/module/openvswitch ]] && echo kernel || echo user)}
export DATAPATH

# --------------------------------------------------------------------------------- stage table
# id|Title|Description|outputs (comma-separated, relative to the project root)
STAGES=(
  "preflight|Preflight checks|Python packages, datasets, Docker and the dashboard port|"
  "dashboard|Start the dashboard|Web app on :$PORT: offline results, live IDS and this progress page|"
  "phase1|Phase 1: binary GNN on CICIDS2017|Clean, RF/XGBoost baselines, 5 s graph snapshots, edge-aware GAT, evaluation|results/baseline_metrics.json,results/gnn_metrics.json,results/gnn_classification_report.json,results/gnn_confusion_matrix.png"
  "clean_insdn|Phase 2: clean InSDN|Keeps the attack type and maps InSDN 'Normal' to benign|data/insdn/cleaned/cleaning_report.json"
  "phase2_train|Phase 2: graphs, baselines, GNN, export|Host graphs with OpenFlow features, fair RF/XGBoost baselines, multi-task GAT$([[ $QUICK == 1 ]] || echo ', ablations'), TorchScript export|data/graphs/insdn_v2.pt,results/phase2/baselines_v2.json,results/phase2/gnn_v2_gat.json,models/gat_ids_insdn_only.pt,models/gat_ids.pt"
  "tests|Tests|pytest: preprocessing, graph builder, Mininet tools, full live path|"
  "replay|Replay through the live IDS|Held-out InSDN flows sent to the running IDS as OpenFlow stats every 2 s (watch /live)|results/phase2/replay_report.json"
  "latency|Latency benchmark|GNN inference time per window size, CPU and GPU|results/phase2/latency_report.json,results/phase2/latency_histogram.png"
  "live_demo|Live SDN demo|Docker lab: os-ken controller, Mininet, attack scenarios, automatic blocking (watch /live)|$SESSION/mitigation_report.json,$SESSION/transfer_eval.json"
  "report|Results report|Collects every Phase 2 result into one Markdown report|results/phase2/final_results.md"
)
declare -A STAGE_LOG
stage_args=()
i=0
for spec in "${STAGES[@]}"; do
  i=$((i + 1))
  id=${spec%%|*}
  STAGE_LOG[$id]=$(printf '%s/%02d_%s.log' "$LOG_DIR" "$i" "$id")
  stage_args+=(--stage "$spec")
done
mkdir -p "$LOG_DIR"

status() { "$PY" scripts/pipeline_status.py "$STATUS" "$@"; }
status init --run-id "$RUN_ID" --log-dir "$LOG_DIR" --options="$*" "${stage_args[@]}" || exit 1
status set dashboard_url "$DASH_URL"
status set session "$SESSION"

banner() { printf '\n\033[1;36m==> [%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }

# run_stage ID FUNCTION: output goes to the terminal and to the stage log (shown on /pipeline).
run_stage() {
  local id=$1 fn=$2 rc
  status start "$id"
  banner "$id: ${fn#stage_}"
  "$fn" > >(tee -a "${STAGE_LOG[$id]}") 2>&1
  rc=$?
  sleep 0.2  # let tee flush before the stage is marked finished
  if [[ $rc -eq 0 ]]; then status done "$id"; else status fail "$id" --note="exit code $rc"; fi
  return $rc
}

skip() {  # skip ID REASON
  status skip "$1" --note="$2"
  banner "$1: skipped ($2)"
  echo "skipped: $2" >> "${STAGE_LOG[$1]}"
}

skip_pending() {  # after a critical failure
  local id
  for id in "${!STAGE_LOG[@]}"; do
    "$PY" -c "import json,sys; s=json.load(open('$STATUS')); sys.exit(0 if any(x['id']=='$id' and x['state']=='pending' for x in s['stages']) else 1)" \
      && status skip "$id" --note="$1"
  done
}

# ------------------------------------------------------------------------------------ dashboard
stop_dashboard() {
  if [[ -n $DASH_PID ]]; then
    kill "$DASH_PID" 2> /dev/null
    wait "$DASH_PID" 2> /dev/null
    DASH_PID=""
  fi
}

# start_dashboard [SESSION]: (re)start the web app; with a session, the IDS logs and records flows into it
start_dashboard() {
  local session=${1:-} log=$ROOT/$LOG_DIR/dashboard.log
  stop_dashboard
  (
    cd web || exit 1
    if [[ -n $session ]]; then
      export IDS_LOG_DIR=$ROOT/$session/logs IDS_RECORD_FLOWS=$ROOT/$session/flows.csv
    else
      export IDS_LOG_DIR=$ROOT/$LOG_DIR/ids
    fi
    exec "$PY" -m uvicorn app:app --host 127.0.0.1 --port "$PORT"
  ) >> "$log" 2>&1 &
  DASH_PID=$!
  for _ in $(seq 1 90); do
    curl -sf "$DASH_URL/api/health" > /dev/null && { echo "Dashboard up: $DASH_URL (pid $DASH_PID)"; return 0; }
    kill -0 "$DASH_PID" 2> /dev/null || break
    sleep 1
  done
  echo "Dashboard failed to start; last lines of $log:"
  tail -n 20 "$log"
  return 1
}

live_ids_ready() { curl -s "$DASH_URL/api/health" | grep -q '"live_ids":true'; }

FINISHED=0
on_interrupt() {
  echo
  if [[ $FINISHED == 0 ]]; then
    echo "Interrupted."
    status finish interrupted --note="stopped with Ctrl+C" 2> /dev/null
  fi
  stop_dashboard
  exit 130
}
trap on_interrupt INT TERM

# --------------------------------------------------------------------------------------- stages
stage_preflight() (
  set -e
  echo "Run $RUN_ID, logs in $LOG_DIR"
  echo "Python: $PY ($("$PY" --version 2>&1))"
  "$PY" - << 'EOF'
import importlib
missing = []
for mod in ("torch", "torch_geometric", "pandas", "sklearn", "fastapi", "uvicorn", "yaml", "pytest"):
    try:
        importlib.import_module(mod)
    except ImportError:
        missing.append(mod)
if missing:
    raise SystemExit("Missing Python packages: " + ", ".join(missing)
                     + "\nInstall with: pip install -r requirements.txt -r web/requirements.txt")
import torch
print(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
EOF
  if has_files "data/insdn/raw/*.csv"; then echo "InSDN raw CSVs: found"
  elif has_files "data/insdn/cleaned/*_cleaned.csv"; then echo "InSDN raw CSVs: missing, cleaned CSVs found (cleaning will be skipped)"
  else echo "InSDN data missing: put the CSVs in data/insdn/raw/"; exit 1; fi
  if has_files "data/cicids2017/raw/*.csv"; then echo "CICIDS2017 raw CSVs: found"
  elif has_files "data/cicids2017/cleaned/*_cleaned.csv"; then echo "CICIDS2017: cleaned CSVs found (Phase 1 starts after cleaning)"
  else echo "CICIDS2017: no data (Phase 1 will be skipped)"; fi
  if [[ $SKIP_TRAINING == 1 && ! -f models/gat_ids.pt ]]; then
    echo "--skip-training needs an exported model (models/gat_ids.pt); run without it once."; exit 1
  fi
  if [[ $SKIP_LIVE == 1 ]]; then echo "Live demo: skipped (--skip-live)"
  elif [[ $DOCKER_OK == 0 ]]; then echo "Live demo: Docker not usable by this user (live demo will be skipped)"
  else
    echo "Docker: OK; lab image: $(docker image inspect gnn-ids-lab > /dev/null 2>&1 && echo present || echo 'missing (will be built)')"
    echo "Open vSwitch datapath: $DATAPATH"
  fi
  echo "Dashboard port $PORT: free"
)

stage_dashboard() {
  start_dashboard || return 1
  live_ids_ready && echo "Live IDS: loaded models/gat_ids.pt" \
    || echo "Live IDS: not loaded yet (no exported model); it starts after Phase 2 training"
  echo "Progress: $DASH_URL/pipeline   Offline results: $DASH_URL/   Live monitor: $DASH_URL/live"
}

stage_phase1() (
  set -e
  if has_files "data/cicids2017/raw/*.csv"; then
    echo "[1/5] Cleaning CICIDS2017 raw CSVs"
    "$PY" -m preprocessing.clean_data --input-glob "data/cicids2017/raw/*.csv" \
      --output-dir data/cicids2017/cleaned --label-column Label
  else
    echo "[1/5] No raw CICIDS2017 CSVs; using data/cicids2017/cleaned"
  fi
  echo "[2/5] Baselines (Random Forest / XGBoost)"
  "$PY" -m baselines.train_baselines --input-glob "data/cicids2017/cleaned/*.csv" --label-col Label \
    --metrics-out results/baseline_metrics.json --cm-out results/baseline_confusion_matrix.png
  echo "[3/5] Graph snapshots (5 s windows, 1 s step)"
  "$PY" -m preprocessing.graph_builder --input-glob "data/cicids2017/cleaned/*.csv" \
    --output-path data/graphs/cicids_graphs.pt --label-col Label \
    --window-seconds 5 --step-seconds 1 --max-graphs-per-file 5000
  echo "[4/5] Training the GAT"
  "$PY" -m models.train_gnn --graph-path data/graphs/cicids_graphs.pt \
    --checkpoint-path models/checkpoints/best_gat.pt --metrics-path results/gnn_metrics.json \
    --batch-size 128 --epochs 40
  echo "[5/5] Evaluation"
  "$PY" -m models.evaluate --graph-path data/graphs/cicids_graphs.pt \
    --checkpoint-path models/checkpoints/best_gat.pt \
    --report-out results/gnn_classification_report.json --cm-out results/gnn_confusion_matrix.png
)

stage_clean_insdn() (
  set -e
  "$PY" -m preprocessing.clean_data --input-glob "data/insdn/raw/*.csv" --output-dir data/insdn/cleaned
)

stage_phase2_train() (
  set -e
  PYTHON=$PY bash scripts/run_phase2_training.sh $([[ $QUICK == 1 ]] && echo --skip-ablations)
)

stage_tests() (
  set -e
  "$PY" -m pytest tests -q
)

stage_replay() (
  set -e
  echo "Replaying held-out InSDN flows through $DASH_URL/api/flows in real time (about 5 minutes)."
  echo "Watch $DASH_URL/live"
  "$PY" scripts/replay_flows.py --url "$DASH_URL/api/flows" --api-key "$IDS_API_KEY" --realtime
)

stage_latency() (
  set -e
  "$PY" scripts/benchmark_latency.py
)

stage_live_demo() {
  mkdir -p "$SESSION/runs" "$SESSION/logs"
  echo "Restarting the IDS for session $SESSION (records flows and the mitigation log there)"
  start_dashboard "$SESSION" || return 1
  (
    set -e
    if ! docker image inspect gnn-ids-lab > /dev/null 2>&1; then
      echo "Building the lab image (about 2 minutes)"
      docker build -t gnn-ids-lab -f docker/Dockerfile.sdn-lab docker/
    fi
    echo "Watch $DASH_URL/live while the attacks run"
    PYTHON=$PY USE_RUNNING_IDS=1 SESSION=$SESSION PORT=$PORT bash scripts/run_phase2_demo.sh
  )
}

stage_report() (
  set -e
  "$PY" scripts/make_phase2_report.py
)

# ------------------------------------------------------------------------------------------ run
FAILED=0
finish_and_serve() {
  local state=$1 note=${2:-}
  status finish "$state" --note="$note"
  FINISHED=1
  echo
  "$PY" - "$STATUS" << 'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
mark = {"done": "\033[32mdone   \033[0m", "failed": "\033[31mFAILED \033[0m", "skipped": "\033[33mskipped\033[0m",
        "pending": "pending", "running": "running"}
print("Summary")
for st in s["stages"]:
    took = f"{(st['finished'] - st['started']) / 60:5.1f} min" if st["started"] and st["finished"] else "         "
    print(f"  {mark[st['state']]}  {took}  {st['title']}" + (f"  ({st['note']})" if st["note"] else ""))
EOF
  echo "Logs: $LOG_DIR"
  if [[ $SERVE == 1 && -n $DASH_PID ]]; then
    echo
    echo "Dashboard running (Ctrl+C to stop):"
    echo "  Progress        $DASH_URL/pipeline"
    echo "  Offline results $DASH_URL/"
    echo "  Live monitor    $DASH_URL/live   (API key for manual block/unblock: $IDS_API_KEY)"
    wait "$DASH_PID"
  else
    stop_dashboard
  fi
}

critical_failure() {  # critical_failure STAGE
  skip_pending "stopped: $1 failed"
  finish_and_serve failed "$1 failed"
  exit 1
}

run_stage preflight stage_preflight || critical_failure preflight
run_stage dashboard stage_dashboard || critical_failure dashboard

if [[ $SKIP_PHASE1 == 1 ]]; then skip phase1 "--skip-phase1"
elif [[ $SKIP_TRAINING == 1 ]]; then skip phase1 "--skip-training"
elif ! has_files "data/cicids2017/raw/*.csv" && ! has_files "data/cicids2017/cleaned/*_cleaned.csv"; then
  skip phase1 "no CICIDS2017 data"
else
  run_stage phase1 stage_phase1 || FAILED=1  # Phase 2 does not depend on Phase 1
fi

if [[ $SKIP_TRAINING == 1 ]]; then
  skip clean_insdn "--skip-training"
  skip phase2_train "--skip-training"
else
  if has_files "data/insdn/raw/*.csv"; then
    run_stage clean_insdn stage_clean_insdn || critical_failure clean_insdn
  else
    skip clean_insdn "no raw CSVs; using the cleaned files"
  fi
  run_stage phase2_train stage_phase2_train || critical_failure phase2_train
  echo "Restarting the dashboard so the live IDS loads the exported model"
  start_dashboard >> "${STAGE_LOG[phase2_train]}" 2>&1 || critical_failure phase2_train
fi

run_stage tests stage_tests || FAILED=1

if live_ids_ready; then
  run_stage replay stage_replay || FAILED=1
else
  status start replay
  echo "Live IDS not running (see $LOG_DIR/dashboard.log)" | tee -a "${STAGE_LOG[replay]}"
  status fail replay --note="live IDS not running"
  FAILED=1
fi

run_stage latency stage_latency || FAILED=1

if [[ $SKIP_LIVE == 1 ]]; then skip live_demo "--skip-live"
elif [[ $DOCKER_OK == 0 ]]; then skip live_demo "Docker not available"
else run_stage live_demo stage_live_demo || FAILED=1
fi

run_stage report stage_report || FAILED=1

if [[ $FAILED == 1 ]]; then
  finish_and_serve failed "some stages failed"
  exit 1
fi
finish_and_serve done
