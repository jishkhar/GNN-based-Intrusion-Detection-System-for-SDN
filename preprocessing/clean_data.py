from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from common.config import parse_args_with_config
from preprocessing.labels import binary_label, canonical_class

# Canonical multi-class label written next to the binary `Label` column.
ATTACK_COLUMN = "Attack"


@dataclass
class CleaningStats:
    file_name: str
    rows_before: int
    rows_after: int
    duplicates_removed: int
    missing_after: int
    class_counts: dict | None = None


def normalize_label(value: object) -> int:
    return binary_label(value)


def clean_dataframe(df: pd.DataFrame, label_column: str) -> tuple[pd.DataFrame, CleaningStats]:
    rows_before = len(df)
    df.columns = [c.strip() for c in df.columns]

    if label_column not in df.columns:
        candidate = next((c for c in df.columns if c.lower() == label_column.lower()), None)
        if not candidate:
            raise ValueError(f"Label column '{label_column}' not found. Columns: {list(df.columns)[:10]}...")
        label_column = candidate

    df = df.replace([np.inf, -np.inf], np.nan)

    duplicates_removed = int(df.duplicated().sum())
    df = df.drop_duplicates()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        median = df[col].median() if not df[col].dropna().empty else 0.0
        df[col] = df[col].fillna(median)

    # Fill remaining non-numeric nulls.
    for col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].fillna("unknown")

    df[ATTACK_COLUMN] = df[label_column].map(canonical_class)
    df[label_column] = df[label_column].map(normalize_label).astype(int)
    missing_after = int(df.isna().sum().sum())

    stats = CleaningStats(
        file_name="",
        rows_before=rows_before,
        rows_after=len(df),
        duplicates_removed=duplicates_removed,
        missing_after=missing_after,
        class_counts={k: int(v) for k, v in df[ATTACK_COLUMN].value_counts().items()},
    )
    return df, stats


def run(input_glob: str, output_dir: str, label_column: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(input_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {input_glob}")

    all_stats: list[dict] = []
    for file_path in files:
        # latin-1 never fails to decode; some CICIDS2017 label strings are not valid UTF-8.
        df = pd.read_csv(file_path, encoding="latin-1", low_memory=False)
        cleaned, stats = clean_dataframe(df, label_column=label_column)
        stats.file_name = os.path.basename(file_path)

        out_name = os.path.splitext(os.path.basename(file_path))[0] + "_cleaned.csv"
        out_path = os.path.join(output_dir, out_name)
        cleaned.to_csv(out_path, index=False)

        all_stats.append(stats.__dict__)
        print(f"Saved cleaned file: {out_path} | rows: {stats.rows_before} -> {stats.rows_after}")

    stats_path = os.path.join(output_dir, "cleaning_report.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)
    print(f"Cleaning report: {stats_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean CICIDS/InSDN CSV files for Phase 1 MVP.")
    parser.add_argument("--input-glob", required=True, help="Input CSV glob, e.g. data/cicids2017/raw/*.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for cleaned CSVs")
    parser.add_argument("--label-column", default="Label", help="Label column name")
    args, _ = parse_args_with_config(parser, "clean")
    run(args.input_glob, args.output_dir, args.label_column)


if __name__ == "__main__":
    main()
