#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.linalg import LinAlgWarning
from scipy.stats import spearmanr

try:
    from scipy.stats import ConstantInputWarning
except Exception:
    ConstantInputWarning = None

from sklearn.linear_model import RidgeCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=LinAlgWarning)
if ConstantInputWarning is not None:
    warnings.filterwarnings("ignore", category=ConstantInputWarning)


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_DIR = SCRIPT_DIR.parent
DATA_DIR = TASK_DIR / "data"
RESULT_DIR = TASK_DIR / "results" / "evo2"

DEFAULT_MUTANTS_PATH = DATA_DIR / "mutants_sequences.tsv"
DEFAULT_PRIMARY_LAYER = "blocks.26"

ALL_STRATEGIES = ['mean']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Evo2 embeddings on infA mutation fitness prediction")
    parser.add_argument("--results_dir", type=Path, default=None, help="Model output directory; defaults to results/<model_key>")
    parser.add_argument('--model_name', default='evo2_7b', help='Evo2 model name', choices=['evo2_7b', 'evo2_1b_base'])
    parser.add_argument("--model_path", type=Path, default=os.environ.get("EVO2_MODEL_PATH"), help="Optional local Evo2 checkpoint passed as local_path")
    parser.add_argument("--model_repo", type=Path, default=os.environ.get("EVO2_REPO"), help="Optional path to the baseline source repository")
    parser.add_argument("--device", default="cuda:0", help='Device for token tensors, e.g. "cuda:0"')
    parser.add_argument('--primary_layer', default=DEFAULT_PRIMARY_LAYER, choices=['blocks.15', 'blocks.26'], help='Representation layer used for the selected Evo2 model.')
    parser.add_argument("--mutants_path", type=Path, default=DEFAULT_MUTANTS_PATH)
    parser.add_argument("--target_col", default="fitness")
    parser.add_argument('--strategies', default=','.join(ALL_STRATEGIES), choices=['mean'], help='Pooling strategy used in the paper.')
    parser.add_argument('--outer_splits', type=int, default=5, choices=[5])
    parser.add_argument("--n_jobs", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all")
    parser.add_argument("--sample_mode", choices=["head", "random"], default="head")
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument('--use_standard_scaler', type=int, default=1, choices=[1], help='Standardize embeddings within each training fold.')
    parser.add_argument("--allow_download", action="store_true", help="Allow model downloads; otherwise use the local Hugging Face cache")
    parser.add_argument("--progress_every", type=int, default=200)
    parser.add_argument("--force", action="store_true", help="Replace existing metrics and OOF predictions.")
    parser.add_argument("--reuse_cached_features", action="store_true")
    parser.add_argument(
        "--run_mode",
        choices=["full", "encode_only", "ridge_only"],
        default="full",
        help="full: encode + RidgeCV; encode_only: only save feature cache; ridge_only: only load cache and run RidgeCV.",
    )
    parser.add_argument("--feature_cache_dir", type=Path, default=None)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--output_csv", type=Path, default=None)
    parser.add_argument('--alpha_log_min', type=float, default=-6.0, choices=[-6.0])
    parser.add_argument('--alpha_log_max', type=float, default=6.0, choices=[6.0])
    parser.add_argument('--alpha_num', type=int, default=500, choices=[500])
    args = parser.parse_args()
    layer = "blocks.15" if args.model_name == "evo2_1b_base" else "blocks.26"
    if args.primary_layer != layer:
        parser.error(f"The paper protocol uses {layer} for {args.model_name}.")
    return args


def parse_strategies(raw: str) -> list[str]:
    if raw != ALL_STRATEGIES[0]:
        raise ValueError(f"The paper protocol uses {ALL_STRATEGIES[0]} pooling.")
    return list(ALL_STRATEGIES)


def safe_tag(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw)


def sample_tag(max_samples: int, sample_mode: str, sample_seed: int) -> str:
    if max_samples <= 0:
        return "all"
    if sample_mode == "random":
        return f"random_{max_samples}_seed{sample_seed}"
    return f"head_{max_samples}"


def run_tag(args: argparse.Namespace, strategies: list[str]) -> str:
    strategy_tag = "-".join(strategies)
    return (
        f"{args.model_name}__layer-{safe_tag(args.primary_layer)}__strat-{strategy_tag}"
        f"__samples-{sample_tag(args.max_samples, args.sample_mode, args.sample_seed)}"
        f"__cv{args.outer_splits}"
    )


def prepare_data(mutants_path: Path, target_col: str, max_samples: int, sample_mode: str, sample_seed: int) -> pd.DataFrame:
    df = pd.read_csv(mutants_path, sep="\t")
    required = {"seq", target_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {mutants_path}: {sorted(missing)}")

    df = df.dropna(subset=["seq", target_col]).copy()
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    if max_samples and max_samples > 0:
        n = min(max_samples, len(df))
        if sample_mode == "random":
            df = df.sample(n=n, random_state=sample_seed).reset_index(drop=True)
        else:
            df = df.head(n).copy()
    return df


def pool_mean(embeddings: dict, primary_layer: str, **_kw) -> np.ndarray:
    emb = embeddings[primary_layer]
    return emb[0].mean(dim=0).cpu().float().numpy()


def _spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rho, _ = spearmanr(y_pred, y_true, nan_policy="omit")
    return 0.0 if np.isnan(rho) else float(rho)


def evaluate_with_ridge(
    x_array: np.ndarray,
    y_array: np.ndarray,
    outer_splits: int,
    n_jobs: int,
    alpha_log_min: float,
    alpha_log_max: float,
    alpha_num: int,
    use_standard_scaler: bool,
) -> tuple[dict, np.ndarray]:
    if not use_standard_scaler:
        raise ValueError("The paper protocol requires StandardScaler.")
    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=42)
    alphas = np.logspace(alpha_log_min, alpha_log_max, alpha_num)
    model = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        RidgeCV(alphas=alphas),
    )
    cv_results = cross_validate(
        model,
        x_array,
        y_array,
        cv=outer_cv,
        scoring={"spearman": make_scorer(_spearman_corr, greater_is_better=True)},
        n_jobs=n_jobs,
        return_estimator=True,
    )
    y_pred_oof = np.empty_like(y_array)
    for estimator, (_, test_idx) in zip(cv_results["estimator"], outer_cv.split(x_array, y_array)):
        y_pred_oof[test_idx] = estimator.predict(x_array[test_idx])
    metrics = {
        "mean_spearman": float(np.mean(cv_results["test_spearman"])),
        "std_spearman": float(np.std(cv_results["test_spearman"])),
        "spearman_folds": [float(value) for value in cv_results["test_spearman"]],
        "selected_alphas": [float(estimator.named_steps["ridgecv"].alpha_) for estimator in cv_results["estimator"]],
    }
    return metrics, y_pred_oof


def cache_paths(cache_dir: Path, strategies: list[str]) -> tuple[Path, dict[str, Path]]:
    y_path = cache_dir / "targets.npy"
    x_paths = {strategy: cache_dir / f"X_{strategy}.npy" for strategy in strategies}
    return y_path, x_paths


def can_reuse_cache(cache_dir: Path, strategies: list[str]) -> bool:
    y_path, x_paths = cache_paths(cache_dir, strategies)
    if not y_path.exists():
        return False
    return all(path.exists() for path in x_paths.values())


def save_cache(cache_dir: Path, strategies: list[str], x_arrays: dict[str, np.ndarray], y_array: np.ndarray, df: pd.DataFrame) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    y_path, x_paths = cache_paths(cache_dir, strategies)
    np.save(y_path, y_array)
    np.save(cache_dir / "sequences.npy", df["seq"].astype(str).to_numpy(dtype=str))
    for strategy in strategies:
        np.save(x_paths[strategy], x_arrays[strategy])


def cache_matches_data(cache_dir: Path, df: pd.DataFrame, target_col: str, y_array: np.ndarray) -> bool:
    sequence_path = cache_dir / "sequences.npy"
    if not sequence_path.is_file():
        return False
    cached_sequences = np.load(sequence_path, allow_pickle=False)
    expected_sequences = df["seq"].astype(str).to_numpy(dtype=str)
    expected_targets = df[target_col].to_numpy(dtype=np.float32)
    return np.array_equal(cached_sequences, expected_sequences) and np.array_equal(y_array, expected_targets)


def load_cache(cache_dir: Path, strategies: list[str]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    y_path, x_paths = cache_paths(cache_dir, strategies)
    y_array = np.load(y_path)
    x_arrays = {strategy: np.load(x_paths[strategy]) for strategy in strategies}
    return x_arrays, y_array


def load_model(args: argparse.Namespace):
    if args.model_repo is not None:
        model_repo = args.model_repo.expanduser().resolve()
        if not model_repo.is_dir():
            raise FileNotFoundError(f"Baseline repository not found: {model_repo}")
        sys.path.insert(0, str(model_repo))
    if not args.allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
    from evo2 import Evo2

    print(f"Loading Evo2 model: {args.model_name}")
    if args.model_path is not None:
        checkpoint_path = args.model_path.expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Evo2 checkpoint not found: {checkpoint_path}")
        model = Evo2(args.model_name, local_path=str(checkpoint_path))
    else:
        model = Evo2(args.model_name)
    print(f"Model ready: {args.model_name}, layer={args.primary_layer}, device={args.device}")
    return model


def _run_forward(model, seq: str, layer_list: list[str], device: str) -> dict:
    input_ids = torch.tensor(
        model.tokenizer.tokenize(seq),
        dtype=torch.int,
    ).unsqueeze(0).to(device)
    with torch.no_grad():
        _, embeddings = model(
            input_ids,
            return_embeddings=True,
            layer_names=layer_list,
        )
    return embeddings


def extract_features(
    model,
    args: argparse.Namespace,
    strategies: list[str],
    df: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    features = {strategy: [] for strategy in strategies}
    targets: list[float] = []
    total = len(df)
    progress_every = max(1, args.progress_every)
    for index, row in enumerate(df.itertuples(index=False), start=1):
        embeddings = _run_forward(model, getattr(row, "seq").upper(), [args.primary_layer], args.device)
        for strategy in strategies:
            features[strategy].append(pool_mean(embeddings, args.primary_layer))
        targets.append(float(getattr(row, args.target_col)))
        if index % progress_every == 0 or index == total:
            print(f"Embedded {index}/{total}")
    return (
        {strategy: np.stack(features[strategy], axis=0) for strategy in strategies},
        np.asarray(targets, dtype=np.float32),
    )


def main() -> None:
    args = parse_args()
    if args.results_dir is None:
        model_key = "evo2_1b" if args.model_name == "evo2_1b_base" else "evo2"
        args.results_dir = TASK_DIR / "results" / model_key
    args.results_dir = args.results_dir.resolve()
    args.mutants_path = args.mutants_path.resolve()

    if not args.mutants_path.is_file():
        raise FileNotFoundError(f"Mutants TSV not found: {args.mutants_path}")
    if args.run_mode != "encode_only" and args.outer_splits < 2:
        raise ValueError("--outer_splits must be at least 2")

    strategies = parse_strategies(args.strategies)
    df = prepare_data(args.mutants_path, args.target_col, args.max_samples, args.sample_mode, args.sample_seed)
    if df.empty:
        raise ValueError("No valid samples after filtering")
    if args.run_mode != "encode_only" and len(df) < args.outer_splits:
        raise ValueError(f"n={len(df)} < outer_splits={args.outer_splits}")

    run_dir = args.results_dir
    cache_dir = args.feature_cache_dir.resolve() if args.feature_cache_dir else run_dir / "pooling_cache"
    output_json = args.output_json.resolve() if args.output_json else run_dir / "pooling_compare_results.json"
    output_csv = args.output_csv.resolve() if args.output_csv else run_dir / "pooling_compare_results.csv"
    predictions_dir = run_dir
    if args.run_mode != "encode_only" and not args.force:
        protected = [output_json, output_csv, *predictions_dir.glob("oof_predictions__*.tsv")]
        existing = next((path for path in protected if path.exists()), None)
        if existing is not None:
            raise FileExistsError(f"Output already exists: {existing}; use --force to replace it.")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run dir: {run_dir}")
    print(f"Mutants path: {args.mutants_path}")
    print(f"Model: {args.model_name}")
    print(f"Primary layer: {args.primary_layer}")
    print(f"Strategies: {strategies}")
    print(f"Total samples: {len(df)}")
    print(f"Sampling: mode={args.sample_mode}, seed={args.sample_seed}, max_samples={args.max_samples}")
    print(f"Use StandardScaler: {bool(args.use_standard_scaler)}")
    print(f"Run mode: {args.run_mode}")

    model_holder = {}

    def get_model():
        if "model" not in model_holder:
            model_holder["model"] = load_model(args)
        return model_holder["model"]

    run_encode = args.run_mode in {"full", "encode_only"}
    run_ridge = args.run_mode in {"full", "ridge_only"}
    cache_used = False
    cache_action = "none"
    x_arrays: dict[str, np.ndarray] = {}
    y_array: np.ndarray | None = None

    should_try_cache = args.run_mode == "ridge_only" or args.reuse_cached_features
    if should_try_cache and can_reuse_cache(cache_dir, strategies):
        x_arrays, y_array = load_cache(cache_dir, strategies)
        if all(x_arrays[strategy].ndim == 2 and x_arrays[strategy].shape[0] == len(df) for strategy in strategies) and cache_matches_data(cache_dir, df, args.target_col, y_array):
            cache_used = True
            cache_action = "reused"
            print(f"Reused cached features from: {cache_dir}")
        else:
            print("Cache sequence order, targets, or feature shapes do not match the input; cache rejected.")
            x_arrays = {}
            y_array = None

    if y_array is None or not x_arrays:
        if run_encode:
            x_arrays, y_array = extract_features(get_model(), args, strategies, df)
            save_cache(cache_dir, strategies, x_arrays, y_array, df)
            cache_action = "saved"
            print(f"Saved feature cache to: {cache_dir}")
        else:
            raise FileNotFoundError(f"Missing aligned cache for ridge_only: {cache_dir}. Regenerate features for the current input; sequences.npy and targets.npy must match every input row.")

    cache_status = {
        "cache_dir": str(cache_dir),
        "cache_action": cache_action,
        "cache_used": cache_used,
        "strategy_shapes": {strategy: list(x_arrays[strategy].shape) for strategy in strategies},
        "y_shape": list(y_array.shape),
    }

    if not run_ridge:
        print(f"Feature cache ready: {cache_dir}")
        return

    results: dict[str, dict] = {}
    rows: list[dict] = []
    if run_ridge:
        predictions_dir.mkdir(parents=True, exist_ok=True)
        for strategy in strategies:
            print(f"Evaluating {strategy} ... X={x_arrays[strategy].shape}, y={y_array.shape}")
            metrics, y_pred_oof = evaluate_with_ridge(
                x_array=x_arrays[strategy],
                y_array=y_array,
                outer_splits=args.outer_splits,
                n_jobs=args.n_jobs,
                alpha_log_min=args.alpha_log_min,
                alpha_log_max=args.alpha_log_max,
                alpha_num=args.alpha_num,
                use_standard_scaler=bool(args.use_standard_scaler),
            )
            results[strategy] = metrics

            pred_df = df.loc[:, [col for col in ["pos", "is_wt", "mutation", args.target_col] if col in df.columns]].copy()
            pred_df[f"pred_{args.target_col}_oof"] = y_pred_oof
            pred_df.to_csv(predictions_dir / f"oof_predictions__{strategy}.tsv", sep="\t", index=False)

            rows.append(
                {
                    "strategy": strategy,
                    "n_samples": int(len(df)),
                    "model_name": args.model_name,
                    "primary_layer": args.primary_layer,
                    "use_standard_scaler": bool(args.use_standard_scaler),
                    "cache_used": cache_used,
                    "mean_spearman": metrics["mean_spearman"],
                    "std_spearman": metrics["std_spearman"],
                    "spearman_folds": json.dumps(metrics["spearman_folds"]),
                    "selected_alphas": json.dumps(metrics["selected_alphas"]),
                }
            )
            print(f"RESULT {strategy}: mean_spearman={metrics['mean_spearman']:.4f}")

    ranking = sorted(results.items(), key=lambda kv: kv[1]["mean_spearman"], reverse=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)

    payload = {
        "model_name": args.model_name,
        "device": args.device,
        "primary_layer": args.primary_layer,
        "mutants_path": str(args.mutants_path),
        "results_dir": str(args.results_dir),
        "run_dir": str(run_dir),
        "run_mode": args.run_mode,
        "target_col": args.target_col,
        "strategies": strategies,
        "max_samples": args.max_samples,
        "sample_mode": args.sample_mode,
        "sample_seed": args.sample_seed,
        "use_standard_scaler": bool(args.use_standard_scaler),
        "reuse_cached_features": bool(args.reuse_cached_features),
        "outer_splits": args.outer_splits,
        "n_jobs": args.n_jobs,
        "alpha_log_min": args.alpha_log_min,
        "alpha_log_max": args.alpha_log_max,
        "alpha_num": args.alpha_num,
        "n_samples": int(len(y_array)),
        "cache_status": cache_status,
        "results": results,
        "ranking_by_spearman": [strategy for strategy, _ in ranking],
        "output_csv": str(output_csv),
        "output_predictions_dir": str(predictions_dir),
    }
    output_json.write_text(json.dumps(payload, indent=2))

    print("\n=== Summary ===")
    print(f"Run mode:      {args.run_mode}")
    print(f"Cache action:  {cache_action}")
    if ranking:
        for idx, (strategy, metrics) in enumerate(ranking, start=1):
            print(f"{idx}. {strategy}: mean_spearman={metrics['mean_spearman']:.4f}")
    print(f"Saved CSV:     {output_csv}")
    print(f"Saved JSON:    {output_json}")


if __name__ == "__main__":
    main()
