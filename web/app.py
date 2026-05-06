from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import json
from pathlib import Path
from config import (
    STATIC_DIR,
    GNN_METRICS_FILE,
    BASELINE_METRICS_FILE,
    GNN_CLASSIFICATION_REPORT_FILE,
    API_TITLE,
    API_VERSION,
    API_DESCRIPTION,
)

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Root endpoint - serve index.html
@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


# API Endpoints
@app.get("/api/metrics/gnn")
async def get_gnn_metrics():
    """Get GNN model metrics including training history and test results."""
    try:
        with open(GNN_METRICS_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": f"File not found: {GNN_METRICS_FILE}"}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON in GNN metrics file"}


@app.get("/api/metrics/baseline")
async def get_baseline_metrics():
    """Get baseline model metrics (Random Forest, XGBoost)."""
    try:
        with open(BASELINE_METRICS_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": f"File not found: {BASELINE_METRICS_FILE}"}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON in baseline metrics file"}


@app.get("/api/classification-report")
async def get_classification_report():
    """Get detailed classification report (per-class metrics)."""
    try:
        with open(GNN_CLASSIFICATION_REPORT_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": f"File not found: {GNN_CLASSIFICATION_REPORT_FILE}"}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON in classification report file"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "gnn_metrics": GNN_METRICS_FILE.exists(),
        "baseline_metrics": BASELINE_METRICS_FILE.exists(),
        "classification_report": GNN_CLASSIFICATION_REPORT_FILE.exists(),
    }


@app.get("/api/summary")
async def get_summary():
    """Get a summary of all metrics for quick overview."""
    try:
        with open(GNN_METRICS_FILE, "r") as f:
            gnn = json.load(f)

        with open(BASELINE_METRICS_FILE, "r") as f:
            baseline = json.load(f)

        with open(GNN_CLASSIFICATION_REPORT_FILE, "r") as f:
            classification = json.load(f)

        return {
            "gnn": {
                "accuracy": gnn["test"]["accuracy"],
                "precision": gnn["test"]["precision"],
                "recall": gnn["test"]["recall"],
                "f1": gnn["test"]["f1"],
                "best_epoch": gnn["best_epoch"],
                "best_val_f1": gnn["best_val_f1"],
                "decision_threshold": gnn["decision_threshold"],
            },
            "baseline": {
                "random_forest": baseline["random_forest"],
                "xgboost": baseline["xgboost"],
            },
            "classification": classification,
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3000, reload=True)
