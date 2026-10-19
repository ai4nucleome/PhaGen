#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TASK_DIR = Path(__file__).resolve().parent.parent
MODEL_NAME_TO_KEY = {
    "PhaGen": "phagen",
    "megaDNA": "megadna",
    "NTv2-50M": "ntv2_50m",
    "NTv2-100M": "ntv2_100m",
    "NTv2-250M": "ntv2_250m",
    "NTv2-500M": "ntv2_500m",
    "NTv3-8M": "ntv3_8m",
    "NTv3-100M": "ntv3_100m",
    "NTv3-650M": "ntv3_650m",
    "Evo": "evo",
    "Evo2-1B": "evo2_1b",
    "Evo2": "evo2",
}
FIXED_STRATEGY_BY_MODEL = {
    name: (
        "all_concat_mean" if key in {"phagen", "megadna"}
        else "last_hidden_mean" if key.startswith("ntv") else "mean"
    )
    for name, key in MODEL_NAME_TO_KEY.items()
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", type=Path, default=TASK_DIR / "source_results", help="External original 4,672-row caches; provide this directory or --cache-map.")
    parser.add_argument("--output-results", type=Path, default=TASK_DIR / "results")
    parser.add_argument(
        "--cache-map", type=Path,
        help=(
            "JSON object mapping model keys to source run directories, e.g. "
            "{\"phagen\": \"cache/phagen/run\"}. Relative paths are resolved "
            "against the JSON file's parent. Without a mapping, provide the original "
            "4,672-row cache under --source-results/<model_key>/ directly or in one run directory."
        ),
    )
    parser.add_argument("--raw-table", type=Path, default=TASK_DIR / "data" / "infA-data.csv")
    parser.add_argument("--dedup-data", type=Path, default=TASK_DIR / "data" / "mutants_sequences.tsv")
    parser.add_argument("--models", default="", help="Comma-separated model keys or display names; default: all 12.")
    parser.add_argument('--outer-splits', type=int, default=5, choices=[5])
    parser.add_argument('--seed', type=int, default=42, choices=[42])
    parser.add_argument('--alpha-log-min', type=float, default=-6.0, choices=[-6.0])
    parser.add_argument('--alpha-log-max', type=float, default=6.0, choices=[6.0])
    parser.add_argument('--alpha-num', type=int, default=500, choices=[500])
    parser.add_argument("--n-jobs", type=int, default=5)
    parser.add_argument("--force", action="store_true", help="Replace existing output files; source caches remain read-only.")
    return parser.parse_args()


def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    value, _ = spearmanr(y_pred, y_true, nan_policy="omit")
    return 0.0 if np.isnan(value) else float(value)


def load_dedup_index(
    raw_table: Path, dedup_data: Path,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, int]]:
    raw = pd.read_csv(raw_table)
    missing = sorted({"pos", "codon", "is_wt", "fitness_rich"} - set(raw.columns))
    if missing:
        raise ValueError(f"Missing source infA columns: {missing}")
    raw["is_wt"] = pd.to_numeric(raw["is_wt"], errors="raise").astype(int)
    if not raw["is_wt"].isin([0, 1]).all():
        raise ValueError("Source is_wt must contain only 0 or 1")
    raw_targets = pd.to_numeric(raw["fitness_rich"], errors="raise").to_numpy(dtype=np.float64)
    wt_indices = np.flatnonzero(raw["is_wt"].to_numpy() == 1)
    if len(raw) != 4672 or len(wt_indices) != 73:
        raise ValueError(f"Expected 4,672 source rows and 73 WT rows, got {len(raw):,} and {len(wt_indices):,}")
    if not np.isfinite(raw_targets).all() or not np.all(raw_targets[wt_indices] == 1.0):
        raise ValueError("Source fitness must be finite, with fitness=1 for all WT rows")
    keep_mask = raw["is_wt"].to_numpy() == 0
    keep_mask[wt_indices[0]] = True
    keep_indices = np.flatnonzero(keep_mask)

    metadata = pd.read_csv(dedup_data, sep="\t")
    missing = sorted({"pos", "is_wt", "mutation", "fitness", "seq"} - set(metadata.columns))
    if missing:
        raise ValueError(f"Missing deduplicated infA columns: {missing}")
    if len(metadata) != 4600 or metadata["seq"].nunique() != 4600:
        raise ValueError("Expected exactly 4,600 unique infA DNA sequences")
    metadata["is_wt"] = pd.to_numeric(metadata["is_wt"], errors="raise").astype(int)
    if not metadata["is_wt"].isin([0, 1]).all() or int(metadata["is_wt"].sum()) != 1:
        raise ValueError("Deduplicated infA metadata must contain exactly one WT row")
    dedup_targets = pd.to_numeric(metadata["fitness"], errors="raise").to_numpy(dtype=np.float64)
    if not np.allclose(dedup_targets, raw_targets[keep_indices], rtol=0.0, atol=0.0):
        raise ValueError("Deduplicated fitness is not aligned to the source design table")
    for column in ("pos", "is_wt"):
        actual = pd.to_numeric(metadata[column], errors="raise").to_numpy()
        expected = pd.to_numeric(raw[column], errors="raise").to_numpy()[keep_indices]
        if not np.array_equal(actual, expected):
            raise ValueError(f"Deduplicated {column} is not aligned to the source design table")

    wt_sequence = str(metadata.loc[metadata["is_wt"].eq(1), "seq"].iloc[0])
    if len(wt_sequence) != 219 or any(base not in "ACGT" for base in wt_sequence):
        raise ValueError("Expected the 219-nt A/C/G/T infA WT sequence")
    reconstructed = []
    for row in raw.itertuples(index=False):
        position = int(row.pos)
        codon = str(row.codon).upper().strip()
        if not 1 <= position <= 73 or len(codon) != 3 or any(base not in "ACGT" for base in codon):
            raise ValueError(f"Invalid source codon or position: {codon}, {position}")
        start = (position - 1) * 3
        sequence = wt_sequence[:start] + codon + wt_sequence[start + 3:]
        if bool(row.is_wt) != (sequence == wt_sequence):
            raise ValueError("WT flags disagree with the source codon sequences")
        reconstructed.append(sequence)
    if metadata["seq"].tolist() != [reconstructed[index] for index in keep_indices]:
        raise ValueError("Deduplicated DNA sequence order does not match the source design table")
    stats = {
        "n_source_rows": len(raw),
        "n_source_wt_rows": len(wt_indices),
        "n_unique_rows": len(metadata),
        "n_retained_wt_rows": int(metadata["is_wt"].sum()),
        "n_removed_duplicate_wt_rows": len(wt_indices) - 1,
    }
    return keep_indices, raw_targets, metadata, stats


def evaluate_with_ridge(
    x_array: np.ndarray, y_array: np.ndarray, *, outer_splits: int,
    seed: int, alphas: np.ndarray, n_jobs: int,
) -> tuple[dict[str, Any], np.ndarray]:
    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    model = make_pipeline(StandardScaler(with_mean=True, with_std=True), RidgeCV(alphas=alphas))
    cv_results = cross_validate(
        model, x_array, y_array, cv=outer_cv,
        scoring={"spearman": make_scorer(spearman_corr, greater_is_better=True)},
        n_jobs=n_jobs, return_estimator=True,
    )
    predictions = np.empty_like(y_array)
    for estimator, (_, test_idx) in zip(cv_results["estimator"], outer_cv.split(x_array, y_array)):
        predictions[test_idx] = estimator.predict(x_array[test_idx])
    spearman_folds = [float(value) for value in cv_results["test_spearman"]]
    metrics = {
        "mean_spearman": float(np.mean(spearman_folds)),
        "std_spearman": float(np.std(spearman_folds)),
        "spearman_folds": spearman_folds,
        "selected_alphas": [float(estimator.named_steps["ridgecv"].alpha_) for estimator in cv_results["estimator"]],
    }
    return metrics, predictions


def resolve_feature_path(source_run: Path, strategy: str) -> Path:
    cache_dir = source_run / "pooling_cache"
    for path in (cache_dir / f"X_{strategy}.npy", cache_dir / "features.npy"):
        if path.is_file():
            return path
    raise FileNotFoundError(f"No feature cache found for strategy={strategy} under {source_run}")


def selected_model_keys(raw: str) -> set[str]:
    requested = {item.strip() for item in raw.split(",") if item.strip()}
    if not requested:
        return set(MODEL_NAME_TO_KEY.values())
    resolved = {MODEL_NAME_TO_KEY.get(item, item) for item in requested}
    unknown = resolved - set(MODEL_NAME_TO_KEY.values())
    if unknown:
        raise ValueError(f"Unknown model keys: {sorted(unknown)}")
    return resolved


def fixed_run_specs(source_results: Path, cache_map: Path | None, wanted: set[str]) -> list[dict[str, Any]]:
    mapped: dict[str, Path] = {}
    if cache_map is not None:
        payload = json.loads(cache_map.read_text())
        if not isinstance(payload, dict):
            raise ValueError("--cache-map must contain a JSON object mapping model keys to run paths")
        for key, value in payload.items():
            if key not in MODEL_NAME_TO_KEY.values() or not isinstance(value, str):
                raise ValueError(f"Invalid cache-map entry: {key}")
            path = Path(value).expanduser()
            mapped[key] = (path if path.is_absolute() else cache_map.parent / path).resolve()
    specs = []
    for name, key in MODEL_NAME_TO_KEY.items():
        if key not in wanted:
            continue
        strategy = FIXED_STRATEGY_BY_MODEL[name]
        if key in mapped:
            source_run = mapped[key]
        else:
            candidates = []
            model_dir = source_results / key
            target_paths = [model_dir / "pooling_cache/targets.npy", *sorted(model_dir.glob("*/pooling_cache/targets.npy"))]
            for target_path in target_paths:
                if not target_path.is_file():
                    continue
                run = target_path.parent.parent
                try:
                    resolve_feature_path(run, strategy)
                except FileNotFoundError:
                    continue
                candidates.append(run)
            if len(candidates) != 1:
                raise ValueError(f"Expected one {key}/{strategy} source run, found {len(candidates)}. Select it explicitly with --cache-map.")
            source_run = candidates[0]
        feature_path = resolve_feature_path(source_run, strategy)
        target_path = source_run / "pooling_cache" / "targets.npy"
        if not target_path.is_file():
            raise FileNotFoundError(target_path)
        specs.append({"model": name, "model_key": key, "strategy": strategy, "source_run": source_run, "feature_path": feature_path, "target_path": target_path})
    return specs


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, list) else value for key, value in row.items()})


def main() -> None:
    args = parse_args()
    if not 2 <= args.outer_splits <= 4600 or args.alpha_num < 2 or args.alpha_log_min >= args.alpha_log_max:
        raise ValueError("Invalid outer fold count or alpha-grid bounds")
    source_results = args.source_results.resolve()
    output_results = args.output_results.resolve()
    if output_results == source_results:
        raise ValueError("Source and output results directories must differ")
    keep_indices, raw_targets, metadata, dedup_stats = load_dedup_index(args.raw_table, args.dedup_data)
    alphas = np.logspace(args.alpha_log_min, args.alpha_log_max, args.alpha_num)
    specs = fixed_run_specs(source_results, args.cache_map.resolve() if args.cache_map else None, selected_model_keys(args.models))
    summary_csv = output_results / "fixed_strategy_summary.csv"
    audit_json = output_results / "dedup_audit.json"
    for spec in specs:
        spec["output_run"] = (output_results / spec["model_key"]).resolve()
        for source_spec in specs:
            source_run = source_spec["source_run"]
            if output_results == source_run or source_run in output_results.parents:
                raise ValueError("Output results must be outside every source cache run")
            if (
                spec["output_run"] == source_run
                or source_run in spec["output_run"].parents
                or spec["output_run"] in source_run.parents
            ):
                raise ValueError("Output run must not overwrite a source cache run")
        protected = [
            spec["output_run"] / "pooling_compare_results.csv",
            spec["output_run"] / "pooling_compare_results.json",
            *spec["output_run"].glob("oof_predictions__*.tsv"),
        ]
        existing = next((path for path in protected if path.exists()), None)
        if existing is not None and not args.force:
            raise FileExistsError(f"Output already exists: {existing}; pass --force")
    for path in (summary_csv, audit_json):
        if path.exists() and not args.force:
            raise FileExistsError(f"Output already exists: {path}; pass --force")
    protocol = {
        "dedup_policy": "retain all mutant rows and the first of 73 identical WT rows",
        **dedup_stats,
        "use_standard_scaler": True,
        "outer_splits": args.outer_splits, "seed": args.seed,
        "alpha_log_min": args.alpha_log_min, "alpha_log_max": args.alpha_log_max,
        "alpha_num": args.alpha_num, "n_jobs": args.n_jobs,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "pandas_version": pd.__version__, "scipy_version": scipy.__version__,
        "scikit_learn_version": sklearn.__version__,
    }
    summary_rows = []
    for index, spec in enumerate(specs, start=1):
        strategy = spec["strategy"]
        x_source = np.load(spec["feature_path"], mmap_mode="r", allow_pickle=False)
        y_source = np.load(spec["target_path"], mmap_mode="r", allow_pickle=False)
        if x_source.ndim != 2 or x_source.shape[0] != len(raw_targets) or y_source.shape != raw_targets.shape:
            raise ValueError(f"Cache shape mismatch for {spec['model_key']}: X={x_source.shape}, y={y_source.shape}")
        if not np.allclose(y_source, raw_targets, rtol=0.0, atol=1e-7):
            raise ValueError(f"Target cache is not aligned for {spec['model_key']}")
        x_dedup = np.asarray(x_source[keep_indices], dtype=np.float32)
        y_dedup = np.asarray(y_source[keep_indices], dtype=np.float32)
        if not np.isfinite(x_dedup).all():
            raise ValueError(f"Non-finite features for {spec['model_key']}")
        print(f"[{index:02d}/{len(specs):02d}] {spec['model_key']}/{strategy}", flush=True)
        metrics, predictions = evaluate_with_ridge(x_dedup, y_dedup, outer_splits=args.outer_splits, seed=args.seed, alphas=alphas, n_jobs=args.n_jobs)
        output_run = spec["output_run"]
        predictions_dir = output_run
        predictions_dir.mkdir(parents=True, exist_ok=True)
        pred_frame = metadata[["pos", "is_wt", "mutation", "fitness"]].copy()
        pred_frame["pred_fitness_oof"] = predictions
        pred_frame.to_csv(predictions_dir / f"oof_predictions__{strategy}.tsv", sep="\t", index=False)
        row = {"model": spec["model"], "model_key": spec["model_key"], "strategy": strategy, "n_samples": len(y_dedup), **metrics, "run_dir": str(output_run)}
        summary_rows.append(row)
        write_csv(output_run / "pooling_compare_results.csv", [{**row, "use_standard_scaler": True, "cache_used": True}])
        (output_run / "pooling_compare_results.json").write_text(json.dumps({**protocol, "model_key": spec["model_key"], "source_run": str(spec["source_run"]), "source_feature_cache": str(spec["feature_path"]), "strategies": [strategy], "results": {strategy: metrics}}, indent=2) + "\n")
        print(f"  mean_spearman={metrics['mean_spearman']:.6f}", flush=True)
    summary_rows.sort(key=lambda row: row["mean_spearman"], reverse=True)
    write_csv(summary_csv, summary_rows)
    audit_json.write_text(json.dumps({**protocol, "source_results": str(source_results), "output_results": str(output_results), "ranking_by_mean_spearman": [row["model"] for row in summary_rows], "fixed_strategy_summary_csv": str(summary_csv)}, indent=2) + "\n")
    print(f"Wrote: {summary_csv}\nWrote: {audit_json}")


if __name__ == "__main__":
    main()
