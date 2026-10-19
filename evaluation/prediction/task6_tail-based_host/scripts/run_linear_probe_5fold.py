#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from threadpoolctl import threadpool_limits


TASK_DIR = Path(__file__).resolve().parent.parent
MODELS = {
    "phagen": "PhaGen",
    "megadna": "megaDNA",
    "evo": "Evo1-7B",
    "evo2_1b": "Evo2-1B",
    "evo2_7b": "Evo2-7B",
    "ntv2_50m": "NTv2-50M",
    "ntv2_100m": "NTv2-100M",
    "ntv2_250m": "NTv2-250M",
    "ntv2_500m": "NTv2-500M",
    "ntv3_8m": "NTv3-8M",
    "ntv3_100m": "NTv3-100M",
    "ntv3_650m": "NTv3-650M",
}
METRICS = ("macro_f1", "mcc", "accuracy")


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def make_folds(data: pd.DataFrame) -> np.ndarray:
    folds = np.full(len(data), -1, dtype=np.int64)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (_, test) in enumerate(
        splitter.split(np.zeros(len(data)), data["host"], data["genome_id"])
    ):
        folds[test] = fold
    return folds


def validate_folds(data: pd.DataFrame, folds: np.ndarray) -> pd.DataFrame:
    if len(folds) != len(data) or set(folds.tolist()) != set(range(5)):
        raise ValueError("Expected complete five-fold assignments.")
    classes = set(data["host"])
    rows = []
    for fold in range(5):
        train, test = data.loc[folds != fold], data.loc[folds == fold]
        overlaps = {
            field: len(set(train[field]) & set(test[field]))
            for field in ["genome_id", "dna", "protein"]
        }
        if (
            any(overlaps.values())
            or set(train["host"]) != classes
            or set(test["host"]) != classes
        ):
            raise ValueError(
                f"Fold {fold} has sequence/genome overlap or missing host classes."
            )
        rows.append(
            {
                "fold": fold,
                "n_train": len(train),
                "n_test": len(test),
                "train_genomes": train["genome_id"].nunique(),
                "test_genomes": test["genome_id"].nunique(),
                "genome_overlap": overlaps["genome_id"],
                "exact_dna_overlap": overlaps["dna"],
                "exact_protein_overlap": overlaps["protein"],
            }
        )
    return pd.DataFrame(rows)


def find_cache(cache_root: Path, key: str) -> tuple[Path, np.ndarray]:
    directory = cache_root / key
    candidates = sorted(directory.rglob("all_X__*.npy"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one feature array for {key}, found {len(candidates)}."
        )
    feature_path = candidates[0]
    ids = np.load(feature_path.parent / "record_ids.npy", allow_pickle=False)
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    if (
        features.ndim != 2
        or features.shape[0] != 12506
        or features.shape[1] < 1
        or not np.array_equal(ids, np.arange(12506))
    ):
        raise ValueError(f"Expected the aligned 12,506-row embedding inputs for {key}.")
    for start in range(0, len(ids), 1024):
        if not np.isfinite(features[start : start + 1024]).all():
            raise ValueError(f"Non-finite features for {key}.")
    return feature_path, features


def embedding_indices(data: pd.DataFrame, cache_rows: int) -> np.ndarray:
    values = pd.to_numeric(data["embedding_row"], errors="raise").to_numpy()
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or (values >= cache_rows).any()
        or not np.array_equal(values, np.floor(values))
        or len(np.unique(values)) != len(data)
    ):
        raise ValueError(
            "Embedding rows must be unique integers within the feature array."
        )
    return values.astype(np.int64)


def evaluate_features(
    features: np.ndarray, data: pd.DataFrame, folds: np.ndarray, threads: int = 8
) -> tuple[pd.DataFrame, np.ndarray]:
    labels = LabelEncoder().fit_transform(data["host"].astype(str))
    embedding_rows = embedding_indices(data, features.shape[0])
    predictions = np.full(len(data), -1, dtype=np.int64)
    rows = []
    with threadpool_limits(limits=threads):
        for fold in range(5):
            train, test = folds != fold, folds == fold
            estimator = make_pipeline(
                StandardScaler(with_mean=True, with_std=True),
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=42,
                ),
            )
            estimator.fit(features[embedding_rows[train]], labels[train])
            prediction = estimator.predict(features[embedding_rows[test]])
            predictions[test] = prediction
            rows.append(
                {
                    "fold": fold,
                    "n_train": int(train.sum()),
                    "n_test": int(test.sum()),
                    **classification_metrics(labels[test], prediction),
                }
            )
    if (predictions < 0).any():
        raise ValueError("Incomplete held-out predictions.")
    return pd.DataFrame(rows), predictions


def summarize_folds(folds: pd.DataFrame) -> dict[str, float]:
    summary = {}
    for metric in METRICS:
        summary[f"mean_fold_{metric}"] = float(folds[metric].mean())
        summary[f"std_fold_{metric}"] = float(folds[metric].std(ddof=1))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path", type=Path, default=TASK_DIR / "data/benchmark.csv"
    )
    parser.add_argument("--cache-root", type=Path, default=TASK_DIR / "feature_cache")
    parser.add_argument("--results-dir", type=Path, default=TASK_DIR / "results")
    parser.add_argument("--fold-assignments", type=Path)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    data = pd.read_csv(args.data_path, low_memory=False)
    if (
        len(data) != 12236
        or data["genome_id"].nunique() != 2214
        or data["host"].nunique() != 7
    ):
        raise ValueError("Expected the 12,236-CDS, 2,214-genome, seven-host benchmark.")
    if not np.array_equal(data["record_id"].to_numpy(), np.arange(len(data))):
        raise ValueError("Benchmark record IDs must follow the frozen input order.")
    embedding_indices(data, 12506)
    if (data.groupby("genome_id")["host"].nunique() != 1).any():
        raise ValueError("Each source genome must have one host label.")
    if args.threads < 1:
        parser.error("threads must be positive.")
    folds = make_folds(data)
    if args.fold_assignments:
        saved = pd.read_csv(args.fold_assignments, sep="\t")
        for field in ["record_id", "host", "genome_id"]:
            if not np.array_equal(saved[field].to_numpy(), data[field].to_numpy()):
                raise ValueError(
                    f"Saved fold table has different {field} values or order."
                )
        folds = saved["fold"].to_numpy(dtype=np.int64)
    leakage = validate_folds(data, folds)
    if args.validate_only:
        print(
            "Validated 12,236 CDSs, 2,214 genomes, seven hosts and five disjoint folds."
        )
        return
    keys = list(
        dict.fromkeys(key.strip() for key in args.models.split(",") if key.strip())
    )
    if not keys or set(keys) - set(MODELS):
        parser.error("Select known model keys.")
    targets = [
        args.results_dir / key / filename
        for key in keys
        for filename in ["fold_metrics.tsv", "summary.json", "oof_predictions.tsv"]
    ]
    if not args.force:
        for path in targets:
            if path.exists():
                raise FileExistsError(
                    f"Output already exists: {path}; use --force to replace it."
                )
    for key in keys:
        _, features = find_cache(args.cache_root, key)
        fold_metrics, prediction = evaluate_features(
            features, data, folds, args.threads
        )
        fold_metrics.insert(0, "model_key", key)
        fold_metrics.insert(1, "model", MODELS[key])
        summary = {
            "model_key": key,
            "model": MODELS[key],
            "n": len(data),
            "feature_dim": features.shape[1],
            **summarize_folds(fold_metrics),
        }
        directory = args.results_dir / key
        directory.mkdir(parents=True, exist_ok=True)
        fold_metrics.to_csv(directory / "fold_metrics.tsv", sep="\t", index=False)
        predictions = data[
            [
                "record_id",
                "embedding_row",
                "source_index",
                "genome_id",
                "seq_id",
                "host",
            ]
        ].copy()
        predictions["fold"] = folds
        predictions["predicted_host"] = np.sort(data["host"].unique())[prediction]
        predictions["correct"] = predictions["host"] == predictions["predicted_host"]
        predictions.to_csv(directory / "oof_predictions.tsv", sep="\t", index=False)
        (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"{MODELS[key]}: mean-fold macro-F1={summary['mean_fold_macro_f1']:.6f}")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    assignments = data[
        ["record_id", "embedding_row", "source_index", "genome_id", "seq_id", "host"]
    ].copy()
    assignments["fold"] = folds
    assignments.to_csv(args.results_dir / "fold_assignments.tsv", sep="\t", index=False)
    leakage.to_csv(args.results_dir / "fold_leakage_audit.tsv", sep="\t", index=False)
    pd.crosstab(data["host"], folds).to_csv(
        args.results_dir / "fold_class_counts.tsv", sep="\t"
    )
    summaries = [
        json.loads((args.results_dir / key / "summary.json").read_text())
        for key in MODELS
        if (args.results_dir / key / "summary.json").is_file()
    ]
    summary = pd.DataFrame(summaries).drop(
        columns=["mean_fold_macro_f1_rank"], errors="ignore"
    )
    summary = summary.sort_values(
        ["mean_fold_macro_f1", "model"], ascending=[False, True]
    )
    summary.insert(0, "mean_fold_macro_f1_rank", np.arange(1, len(summary) + 1))
    summary.to_csv(args.results_dir / "model_summary.tsv", sep="\t", index=False)
    protocol = {
        "task": "Seven-class host prediction from tail-protein CDSs",
        "n": len(data),
        "n_source_genomes": int(data["genome_id"].nunique()),
        "classes": data["host"].value_counts().sort_index().to_dict(),
        "embedding_input_rows": 12506,
        "evaluation_rows": 12236,
        "split": {
            "type": "StratifiedGroupKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42,
            "group_column": "genome_id",
            "shared_across_models": True,
        },
        "probe": {
            "estimator": "StandardScaler + LogisticRegression",
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 5000,
            "random_state": 42,
        },
        "reported_metrics": list(METRICS),
        "primary_metric": "mean held-out-fold macro_f1",
        "uncertainty": "sample standard deviation across the five held-out folds",
    }
    (args.results_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
