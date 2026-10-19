#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from embedding_utils.encoders import MODEL_SPECS

METRICS = ("macro_f1", "mcc", "accuracy")


def load_data(path):
    table = pd.read_csv(path, sep="\t", low_memory=False, keep_default_na=False)
    if len(table) != 14901 or table["split"].eq("train").sum() != 7822:
        raise ValueError("Expected the 14,901-record paper table with 7,822 training records")
    if set(table["split"]) != {"train", "test"} or table["label"].nunique() != 7:
        raise ValueError("Expected the official two-way split and seven protein classes")
    if table["accession"].duplicated().any() or table["dna_seq"].duplicated().any():
        raise ValueError("Accessions and CDS sequences must be unique")
    if set(table["similarity_threshold"].astype(str)) != {"40"}:
        raise ValueError("Expected the official similarity threshold 40")
    return table


def score_predictions(truth, prediction):
    return {"macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
            "mcc": float(matthews_corrcoef(truth, prediction)),
            "accuracy": float(accuracy_score(truth, prediction))}


def fit_probe(features, table):
    train = table["split"].eq("train").to_numpy()
    test = ~train
    labels = LabelEncoder().fit(table["label"].astype(str))
    y = labels.transform(table["label"])
    estimator = make_pipeline(StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=5000, random_state=42))
    estimator.fit(features[train], y[train])
    predicted = estimator.predict(features[test])
    probabilities = estimator.predict_proba(features[test])
    result = pd.DataFrame({"accession": table.loc[test, "accession"].to_numpy(),
        "y_true": labels.inverse_transform(y[test]), "y_pred": labels.inverse_transform(predicted)})
    for column, label in enumerate(labels.classes_):
        result[f"score__{label}"] = probabilities[:, column]
    return score_predictions(y[test], predicted), result


def load_features(cache_dir, key, table):
    strategy, dimension = MODEL_SPECS[key][1:3]
    if not np.array_equal(np.load(cache_dir / "record_ids.npy"), np.arange(len(table), dtype=np.int64)):
        raise ValueError("Cached record IDs do not match source table order")
    sequences = cache_dir / "sequences.npy"
    if sequences.exists() and not np.array_equal(np.load(sequences), table["dna_seq"].astype(str).to_numpy()):
        raise ValueError("Cached sequences do not match the source table")
    features = np.load(cache_dir / f"all_X__{strategy}.npy", mmap_mode="r")
    if features.shape != (len(table), dimension) or not np.isfinite(features).all():
        raise ValueError(f"Invalid feature array: {features.shape}")
    return features


def write_result(output_dir, key, metrics, predictions, n_train=7822):
    display, strategy, dimension = MODEL_SPECS[key][:3]
    summary = {"model_key": key, "model": display, "strategy": strategy, "feature_dim": dimension,
        "n_train": int(n_train), "n_test": len(predictions), **metrics}
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(output_dir / "metrics.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    predictions.to_csv(output_dir / "predictions.tsv", sep="\t", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=MODEL_SPECS, required=True)
    parser.add_argument("--data-path", type=Path, default=ROOT / "data/processed/phavip_threshold40.tsv")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir or ROOT / "results" / args.model_key
    if any((output_dir / name).exists() for name in ("metrics.csv", "metrics.json", "predictions.tsv")) and not args.force:
        raise FileExistsError(f"Results already exist in {output_dir}; use --force")
    table = load_data(args.data_path)
    features = load_features(args.cache_dir or ROOT / "feature_cache" / args.model_key, args.model_key, table)
    metrics, predictions = fit_probe(features, table)
    summary = write_result(output_dir, args.model_key, metrics, predictions)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
