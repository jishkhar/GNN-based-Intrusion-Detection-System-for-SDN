import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
STATIC_DIR = BASE_DIR / "static"

# API Configuration
API_TITLE = "GNN-based IDS Dashboard API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "API for GNN-based Intrusion Detection System for Software-Defined Networks"

# Server Configuration
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 3000
DEBUG = True

# Result files
GNN_METRICS_FILE = RESULTS_DIR / "gnn_metrics.json"
BASELINE_METRICS_FILE = RESULTS_DIR / "baseline_metrics.json"
GNN_CLASSIFICATION_REPORT_FILE = RESULTS_DIR / "gnn_classification_report.json"
PHASE2_RESULTS_DIR = RESULTS_DIR / "phase2"

# Progress of scripts/run_all.sh (written by scripts/pipeline_status.py)
PIPELINE_DIR = PROJECT_ROOT / "logs" / "pipeline"
PIPELINE_STATUS_FILE = PIPELINE_DIR / "status.json"
