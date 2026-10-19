#!/usr/bin/env python3

from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from common import (
    DEFAULT_DATA_PATH,
    MODEL_LABELS,
    TASK_DIR,
    PROTOCOL,
    METRICS,
    load_cv_df,
    metrics_from_oof,
    dump_json,
    output_reference,
)


def collect(results_dir, data_path, force=False):
    df = load_cv_df(data_path)
    if len(df) != 3200 or int(df["y"].sum()) != 800 or set(df["fold"]) != set(range(5)):
        raise ValueError(
            "Expected the 3,200-window paper benchmark and all five folds."
        )
    targets = [results_dir / "summary.tsv"]
    for model_key in MODEL_LABELS:
        targets += [
            results_dir / model_key / name
            for name in ("summary.json", "fold_metrics.tsv")
        ]
    if not force and any(path.exists() for path in targets):
        raise FileExistsError("Metric tables exist; use --force to replace them.")
    rows = []
    pending = []
    for model_key, label in MODEL_LABELS.items():
        directory = results_dir / model_key
        path = directory / "oof_predictions.tsv"
        oof = pd.read_csv(path, sep="\t", float_precision="round_trip")
        folds, metrics = metrics_from_oof(oof, df)
        strategy = (
            "all_concat_mean"
            if model_key in {"phagen", "megadna"}
            else "last_hidden_mean"
            if model_key.startswith("ntv")
            else "mean"
        )
        payload = {
            "model_key": model_key,
            "model_name": label,
            "strategy": strategy,
            "n_samples": len(df),
            "n_folds": len(folds),
            "protocol": PROTOCOL,
            "metrics": metrics,
            "oof_predictions": output_reference(directory / "oof_predictions.tsv"),
            "fold_metrics": output_reference(directory / "fold_metrics.tsv"),
        }
        row = {
            "model_key": model_key,
            "model_name": label,
            "strategy": strategy,
            "n_samples": len(df),
            "n_folds": len(folds),
            **metrics,
            **{
                f"{metric}_folds": json.dumps(folds[metric].tolist())
                for metric in METRICS
            },
            "oof_predictions": payload["oof_predictions"],
            "fold_metrics": payload["fold_metrics"],
        }
        rows.append(row)
        pending.append((directory, folds, payload))
    for directory, folds, payload in pending:
        folds.to_csv(directory / "fold_metrics.tsv", sep="\t", index=False)
        dump_json(directory / "summary.json", payload)
    pd.DataFrame(rows).to_csv(results_dir / "summary.tsv", sep="\t", index=False)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--results_dir", type=Path, default=TASK_DIR / "results")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(collect(args.results_dir, args.data_path, args.force).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
