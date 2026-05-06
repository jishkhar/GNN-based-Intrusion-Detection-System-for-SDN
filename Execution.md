# Execution Guide (Phase 1 MVP)

This document explains how to run the Phase 1 MVP end-to-end for the GNN-based IDS project.

---

## 1) Prerequisites

- OS: Linux (recommended)
- Python: 3.10+
- Dataset CSVs available locally:
  - CICIDS2017 CSV files
  - InSDN CSV files

---

## 2) Project location

From terminal, move to project root:

```bash
cd /home/nyx/Projects/GNN-based-Intrusion-Detection-System-for-SDN
```

---

## 3) Create and activate environment

### Option A: venv + pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: conda

```bash
conda env create -f environment.yml
conda activate gnn-ids-phase1
```

---

## 4) Place datasets in correct folders

Put raw CSV files here:

- `data/cicids2017/raw/`
- `data/insdn/raw/`

Example expected structure:

- `data/cicids2017/raw/*.csv`
- `data/insdn/raw/*.csv`

Do not place cleaned files manually; the pipeline will generate them.

---

## 5) Quick checks before full run

### 5.1 Check script syntax

```bash
bash -n scripts/run_phase1_pipeline.sh
```

### 5.2 Verify raw files are present

```bash
ls data/cicids2017/raw/*.csv
ls data/insdn/raw/*.csv
```

---

## 6) Run complete Phase 1 pipeline

```bash
bash scripts/run_phase1_pipeline.sh
```

This executes in order:
1. Clean CICIDS raw data
2. Clean InSDN raw data
3. Train/evaluate baselines (RandomForest + XGBoost if installed)
4. Build graph snapshots
5. Train GNN (GAT)
6. Evaluate GNN

---

## 7) Outputs generated

After successful run, check:

- Cleaned data:
  - `data/cicids2017/cleaned/*_cleaned.csv`
  - `data/insdn/cleaned/*_cleaned.csv`
  - `data/*/cleaned/cleaning_report.json`

- Graph dataset:
  - `data/graphs/cicids_graphs.pt`

- Baseline results:
  - `results/baseline_metrics.json`
  - `results/baseline_confusion_matrix.png`

- GNN results:
  - `models/checkpoints/best_gat.pt`
  - `results/gnn_metrics.json`
  - `results/gnn_classification_report.json`
  - `results/gnn_confusion_matrix.png`

---

## 8) Run modules individually (optional)

If debugging is needed, run steps one by one.

### 8.1 Clean CICIDS data

```bash
python -m preprocessing.clean_data \
  --input-glob "data/cicids2017/raw/*.csv" \
  --output-dir "data/cicids2017/cleaned" \
  --label-column "Label"
```

### 8.2 Clean InSDN data

```bash
python -m preprocessing.clean_data \
  --input-glob "data/insdn/raw/*.csv" \
  --output-dir "data/insdn/cleaned" \
  --label-column "Label"
```

### 8.3 Run baselines

```bash
python -m baselines.train_baselines \
  --input-glob "data/cicids2017/cleaned/*.csv" \
  --label-col "Label" \
  --metrics-out "results/baseline_metrics.json" \
  --cm-out "results/baseline_confusion_matrix.png"
```

### 8.4 Build graphs

```bash
python -m preprocessing.graph_builder \
  --input-glob "data/cicids2017/cleaned/*.csv" \
  --output-path "data/graphs/cicids_graphs.pt" \
  --label-col "Label" \
  --window-seconds 5 \
  --step-seconds 1
```

### 8.5 Train GNN

```bash
python -m models.train_gnn \
  --graph-path "data/graphs/cicids_graphs.pt" \
  --checkpoint-path "models/checkpoints/best_gat.pt" \
  --metrics-path "results/gnn_metrics.json" \
  --epochs 40
```

### 8.6 Evaluate GNN

```bash
python -m models.evaluate \
  --graph-path "data/graphs/cicids_graphs.pt" \
  --checkpoint-path "models/checkpoints/best_gat.pt" \
  --report-out "results/gnn_classification_report.json" \
  --cm-out "results/gnn_confusion_matrix.png"
```

---

## 9) Common issues and fixes

- **No files matched error**
  - Ensure raw CSV files exist in `data/*/raw/`.
- **xgboost missing**
  - Install via `pip install xgboost`.
  - Pipeline still runs RandomForest even if XGBoost is unavailable.
- **torch-geometric install issues**
  - Use the same Python version (3.10 recommended).
  - Install `torch` first, then `torch-geometric`.
- **Out-of-memory during training**
  - Reduce epochs or graph volume.
  - Start with fewer raw CSVs for initial validation.

---

## 10) MVP completion checklist

- [ ] Pipeline runs without failure
- [ ] Baseline metrics generated
- [ ] Graph dataset generated
- [ ] GNN model trained and checkpoint saved
- [ ] GNN report + confusion matrix generated
- [ ] Results ready for Phase 1 guide review
