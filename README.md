# GNN-Based Intrusion Detection System for SDN (Phase 1 Codebase)

This repository now includes an executable **Phase 1 MVP codebase** outside the Docs directory.

## What is implemented

- Data cleaning pipeline for CICIDS2017/InSDN CSV files
- Baseline training (RandomForest + optional XGBoost)
- Sliding-window graph snapshot builder for PyTorch Geometric
- GAT-based graph classifier (train + evaluate)
- End-to-end shell script to run the offline MVP pipeline

## Project structure

- preprocessing/
  - clean_data.py
  - feature_extractor.py
  - graph_builder.py
  - graph_dataset.py
- baselines/
  - train_baselines.py
- models/
  - gat_model.py
  - train_gnn.py
  - evaluate.py
- scripts/
  - run_phase1_pipeline.sh

## Setup

1. Install dependencies:
   - `pip install -r requirements.txt`
   - or `conda env create -f environment.yml`
2. Place raw CSV files in:
   - `data/cicids2017/raw/`
   - `data/insdn/raw/`

## Run full Phase 1 pipeline

- `bash scripts/run_phase1_pipeline.sh`

Outputs will be written to `results/` and `models/checkpoints/`.

## Notes

- Phase 1 is offline-only by design (no Mininet/Ryu integration yet).
- Label mapping is binary for MVP: `BENIGN -> 0`, all attacks -> `1`.
- You can extend to multi-class and SDN live integration in Phase 2.
