#!/usr/bin/env python3

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
    matthews_corrcoef,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TASK_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = (
    TASK_DIR / "data" / "processed" / "phage_promoter_dataset2400_65bp.tsv"
)
MODEL_LABELS = {
    "phagen": "PhaGen",
    "megadna": "megaDNA",
    "evo": "Evo1-7B",
    "evo2_1b": "Evo2-1B",
    "evo2": "Evo2-7B",
    "ntv2_50m": "NTv2-50M",
    "ntv2_100m": "NTv2-100M",
    "ntv2_250m": "NTv2-250M",
    "ntv2_500m": "NTv2-500M",
    "ntv3_8m": "NTv3-8M",
    "ntv3_100m": "NTv3-100M",
    "ntv3_650m": "NTv3-650M",
}
NT_MODELS = {
    "ntv2_50m": "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
    "ntv2_100m": "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
    "ntv2_250m": "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
    "ntv2_500m": "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
    "ntv3_8m": "InstaDeepAI/NTv3_8M_pre",
    "ntv3_100m": "InstaDeepAI/NTv3_100M_pre",
    "ntv3_650m": "InstaDeepAI/NTv3_650M_pre",
}
METRICS = ("auroc", "auprc", "mcc")
PROTOCOL = {
    "n_splits": 5,
    "fold_seed": 11,
    "window_length": 65,
    "promoter_similarity_threshold": 0.95,
    "standard_scaler": True,
    "C": 1.0,
    "class_weight": "balanced",
    "solver": "liblinear",
    "max_iter": 2000,
    "classifier_random_state": 42,
    "decision_threshold": 0.5,
    "auprc_definition": "trapezoidal_area_under_precision_recall_curve",
    "reported_metrics": "mean_across_held_out_folds",
    "std_ddof": 1,
}


def parser_for(family):
    parser = argparse.ArgumentParser(
        description=f"Evaluate {family} frozen embeddings on PhagePromoter."
    )
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--results_dir", type=Path, default=None)
    parser.add_argument("--feature_cache_dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--progress_every", type=int, default=200)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Small-run limit; zero uses all rows.",
    )
    parser.add_argument(
        "--run_mode", choices=["full", "encode_only", "probe_only"], default="full"
    )
    parser.add_argument("--reuse_cached_features", action="store_true")
    parser.add_argument(
        "--allow_download",
        action="store_true",
        help="Allow model files to be downloaded; local files/cache are required by default.",
    )
    parser.add_argument(
        "--model_repo",
        type=Path,
        default=None,
        help="Optional local baseline source repository.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Replace existing evaluation outputs."
    )
    return parser


def load_cv_df(path):
    df = pd.read_csv(path, sep="\t", float_precision="round_trip")
    missing = {"sample_id", "accession", "seq", "y", "fold"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["sample_id"] = df["sample_id"].astype(str)
    df["accession"] = df["accession"].astype(str)
    df["seq"] = df["seq"].astype(str).str.upper()
    df["y"] = df["y"].astype(np.int64)
    df["fold"] = df["fold"].astype(np.int64)
    if df["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample IDs")
    if not df["seq"].str.fullmatch("[ACGT]{65}").all():
        raise ValueError("The benchmark requires unambiguous 65 bp DNA windows")
    if not set(df["y"].unique()) <= {0, 1} or not set(df["fold"].unique()) <= set(
        range(5)
    ):
        raise ValueError("Invalid labels or fold assignments")
    for column in ("accession", "cluster_id"):
        if column in df and df.groupby(column)["fold"].nunique().max() > 1:
            raise ValueError(f"{column} crosses folds")
    return df


def compute_metrics(y, score, pred):
    precision, recall, _ = precision_recall_curve(y, score)
    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(auc(recall, precision)),
        "mcc": float(matthews_corrcoef(y, pred)),
    }


def metrics_from_oof(oof, df):
    for key, source_key in (
        ("sample_id", "sample_id"),
        ("accession", "accession"),
        ("fold", "fold"),
        ("y_true", "y"),
    ):
        if not np.array_equal(oof[key].to_numpy(), df[source_key].to_numpy()):
            raise ValueError(f"OOF {key} does not match the benchmark in row order")
    scores = oof["score"].to_numpy(dtype=np.float64)
    predictions = oof["y_pred"].to_numpy(dtype=np.int64)
    if not np.isfinite(scores).all() or not np.array_equal(
        predictions, (scores >= 0.5).astype(np.int64)
    ):
        raise ValueError("OOF probabilities or thresholded predictions are invalid")
    rows = []
    for fold in sorted(df["fold"].unique()):
        mask = df["fold"].to_numpy() == fold
        y = df.loc[mask, "y"].to_numpy()
        if set(y) != {0, 1}:
            raise ValueError(f"Fold {fold} must contain both classes")
        rows.append(
            {
                "fold": int(fold),
                "n_train": int((~mask).sum()),
                "n_valid": int(mask.sum()),
                **compute_metrics(y, scores[mask], predictions[mask]),
            }
        )
    folds = pd.DataFrame(rows)
    summary = {
        f"{stat}_{metric}": float(
            getattr(folds[metric], stat)(**({"ddof": 1} if stat == "std" else {}))
        )
        for metric in METRICS
        for stat in ("mean", "std")
    }
    return folds, summary


def prediction_table(df, pred, score):
    table = pd.DataFrame(
        {
            "sample_id": df["sample_id"].to_numpy(),
            "accession": df["accession"].to_numpy(),
            "fold": df["fold"].to_numpy(),
            "y_true": df["y"].to_numpy(),
            "y_pred": pred,
            "score": score,
        }
    )
    for name in (
        "negative_mode",
        "gc_fraction",
        "negative_gc_abs_delta",
        "negative_motif_score",
    ):
        if name in df:
            table[name] = df[name].to_numpy()
    return table


def fit_logreg_cv(x, df):
    x = np.asarray(x)
    if x.ndim != 2 or len(x) != len(df):
        raise ValueError("Feature matrix does not match the input rows")
    y = df["y"].to_numpy(dtype=np.int64)
    fold_ids = df["fold"].to_numpy(dtype=np.int64)
    oof_score = np.full(len(df), np.nan, dtype=np.float64)
    oof_pred = np.full(len(df), -1, dtype=np.int64)
    for fold in sorted(np.unique(fold_ids)):
        valid = fold_ids == fold
        model = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(
                C=1.0,
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear",
                random_state=42,
            ),
        )
        model.fit(x[~valid], y[~valid])
        score = model.predict_proba(x[valid])[:, 1]
        oof_score[valid] = score
        oof_pred[valid] = (score >= 0.5).astype(np.int64)
    return prediction_table(df, oof_pred, oof_score)


def cache_paths(cache_dir, strategy):
    return cache_dir / f"all_X__{strategy}.npy", cache_dir / "sample_ids.npy"


def dump_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def output_reference(path):
    try:
        return str(path.resolve().relative_to(TASK_DIR.resolve()))
    except ValueError:
        return path.name


def write_results(output_dir, model_key, strategy, oof, df, feature_dim=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    folds, metrics = metrics_from_oof(oof, df)
    oof.to_csv(output_dir / "oof_predictions.tsv", sep="\t", index=False)
    folds.to_csv(output_dir / "fold_metrics.tsv", sep="\t", index=False)
    payload = {
        "model_key": model_key,
        "model_name": MODEL_LABELS[model_key],
        "strategy": strategy,
        "n_samples": len(df),
        "n_folds": len(folds),
        "protocol": PROTOCOL,
        "metrics": metrics,
        "oof_predictions": output_reference(output_dir / "oof_predictions.tsv"),
        "fold_metrics": output_reference(output_dir / "fold_metrics.tsv"),
    }
    if feature_dim is not None:
        payload["feature_dim"] = int(feature_dim)
    dump_json(output_dir / "summary.json", payload)
    return payload


def evaluate(args, model_key, strategy, encoder):
    output_dir = args.results_dir or TASK_DIR / "results" / model_key
    targets = [
        output_dir / name
        for name in ("summary.json", "fold_metrics.tsv", "oof_predictions.tsv")
    ]
    if args.run_mode != "encode_only" and not args.force:
        existing = [str(path) for path in targets if path.exists()]
        if existing:
            raise FileExistsError(
                f"Evaluation output exists: {existing}. Use --force to replace it."
            )
    df = load_cv_df(args.data_path)
    if args.max_samples > 0:
        df = df.head(args.max_samples).reset_index(drop=True)
    cache_dir = args.feature_cache_dir or TASK_DIR / "feature_cache" / model_key
    x_path, ids_path = cache_paths(cache_dir, strategy)
    features = None
    if args.run_mode == "probe_only" or args.reuse_cached_features:
        if x_path.exists() and ids_path.exists():
            cached_ids = np.load(ids_path, allow_pickle=True)
            candidate = np.load(x_path)
            if np.array_equal(
                cached_ids.astype(str), df["sample_id"].to_numpy(dtype=str)
            ) and candidate.shape[0] == len(df):
                features = candidate
        if features is None and args.run_mode == "probe_only":
            raise FileNotFoundError(
                f"Aligned feature cache not found under {cache_dir}"
            )
    if features is None:
        features = encoder(args, df)
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(x_path, features)
        np.save(ids_path, df["sample_id"].to_numpy(dtype=object))
    if args.run_mode == "encode_only":
        return 0
    oof = fit_logreg_cv(features, df)
    payload = write_results(output_dir, model_key, strategy, oof, df, features.shape[1])
    print(json.dumps(payload["metrics"], indent=2))
    return 0
