from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:
    HAS_XGB = False


def _load_csvs(input_glob: str) -> pd.DataFrame:
    files = sorted(glob.glob(input_glob))
    if not files:
        raise FileNotFoundError(f"No files found by {input_glob}")
    frames = [pd.read_csv(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def _prepare_xy(df: pd.DataFrame, label_col: str):
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found")

    y = df[label_col].astype(int)
    drop_cols = {label_col, "Source IP", "Destination IP", "Timestamp"}
    feature_cols = [
        c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not feature_cols:
        raise ValueError("No numeric feature columns found for baseline training")

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, y, feature_cols


def _metrics(y_true, y_pred, y_prob=None):
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        try:
            m["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            m["roc_auc"] = None
    return m


def run(input_glob: str, label_col: str, metrics_out: str, cm_out: str, seed: int = 42) -> None:
    df = _load_csvs(input_glob)
    X, y, feature_cols = _prepare_xy(df, label_col=label_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    rf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=seed,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]

    out = {
        "features": feature_cols,
        "random_forest": _metrics(y_test, rf_pred, rf_prob),
    }

    fig, axes = plt.subplots(1, 2 if HAS_XGB else 1, figsize=(12, 5))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    ConfusionMatrixDisplay.from_predictions(y_test, rf_pred, ax=axes[0], colorbar=False)
    axes[0].set_title("RandomForest")

    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            eval_metric="logloss",
        )
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)
        xgb_prob = xgb.predict_proba(X_test)[:, 1]
        out["xgboost"] = _metrics(y_test, xgb_pred, xgb_prob)
        ConfusionMatrixDisplay.from_predictions(y_test, xgb_pred, ax=axes[1], colorbar=False)
        axes[1].set_title("XGBoost")
    else:
        out["xgboost"] = {"status": "xgboost_not_installed"}

    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    plt.tight_layout()
    plt.savefig(cm_out, dpi=160)
    plt.close(fig)
    print(f"Saved baseline metrics -> {metrics_out}")
    print(f"Saved confusion matrices -> {cm_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate baseline models (RF/XGBoost).")
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--label-col", default="Label")
    parser.add_argument("--metrics-out", default="results/baseline_metrics.json")
    parser.add_argument("--cm-out", default="results/baseline_confusion_matrix.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run(
        input_glob=args.input_glob,
        label_col=args.label_col,
        metrics_out=args.metrics_out,
        cm_out=args.cm_out,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
