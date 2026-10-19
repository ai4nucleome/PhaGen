#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
DATA_DIR = TASK_DIR / "data" / "processed"
RESULT_DIR = TASK_DIR / "results" / "phagen"
DEFAULT_MODEL = "YusenWithR/PhaGen"
DEFAULT_MUTANTS_PATH = DATA_DIR / "mutants_sequences.tsv"

ALL_STRATEGIES = ['all_concat_mean']
DEFAULT_STRATEGIES = ["all_concat_mean"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PhaGen on TEM-1 beta-lactamase task")
    parser.add_argument("--results_dir", type=Path, default=RESULT_DIR, help="Model output directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="PhaGen checkpoint directory or model ID")
    parser.add_argument("--revision", default=None, help="Optional model revision")
    parser.add_argument("--local_files_only", "--local-files-only", action="store_true", default=True, help="Use local checkpoint files or cached model files only (default).")
    parser.add_argument("--allow_download", action="store_false", dest="local_files_only", help="Allow model downloads instead of requiring local files.")
    parser.add_argument("--device", default="cuda:0", help='Inference device, e.g. "cuda:0"')
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--mutants_path", type=Path, default=DEFAULT_MUTANTS_PATH, help="TSV with seq + target column")
    parser.add_argument("--target_col", default="fitness", help="Target column in mutants TSV")
    parser.add_argument('--pad_attention_mode', default='attend_pad', choices=['attend_pad'], help='Attend to the internally padded length.')
    parser.add_argument('--pool_pad_mode', default='exclude_pad', choices=['exclude_pad'], help='Exclude internal padding during pooling.')
    parser.add_argument('--strategies', default=','.join(DEFAULT_STRATEGIES), choices=['all_concat_mean'], help='Pooling strategy used in the paper.')
    parser.add_argument('--outer_splits', type=int, default=5, help='Outer CV folds', choices=[5])
    parser.add_argument("--n_jobs", type=int, default=5, help="Sklearn n_jobs")
    parser.add_argument("--max_samples", type=int, default=0, help="Debug mode: use first N samples (0 means all)")
    parser.add_argument(
        "--sample_mode",
        choices=["head", "random"],
        default="head",
        help='Subsampling mode used when max_samples > 0. "head" keeps original order, "random" samples uniformly.',
    )
    parser.add_argument("--sample_seed", type=int, default=42, help="Random seed for sample_mode=random")
    parser.add_argument('--add_bos_eos', type=int, default=0, choices=[0], help='The paper protocol does not add BOS/EOS tokens.')
    parser.add_argument('--use_standard_scaler', type=int, default=1, choices=[1], help='Standardize embeddings within each training fold.')
    parser.add_argument("--progress_every", type=int, default=500, help="Print progress every N sequences")
    parser.add_argument("--reuse_cached_features", action="store_true", help="Skip model inference and reuse cached .npy features if present")
    parser.add_argument(
        "--run_mode",
        choices=["full", "encode_only", "ridge_only"],
        default="full",
        help="full: encode + RidgeCV; encode_only: only save feature cache; ridge_only: only load cache and run RidgeCV.",
    )
    parser.add_argument("--feature_cache_dir", type=Path, default=None, help="Optional directory for cached feature arrays")
    parser.add_argument("--output_json", type=Path, default=None, help="Optional path to save metrics JSON")
    parser.add_argument("--output_csv", type=Path, default=None, help="Optional path to save metrics CSV")
    parser.add_argument('--alpha_log_min', type=float, default=-6.0, help='RidgeCV alpha log10 min', choices=[-6.0])
    parser.add_argument('--alpha_log_max', type=float, default=6.0, help='RidgeCV alpha log10 max', choices=[6.0])
    parser.add_argument('--alpha_num', type=int, default=500, help='RidgeCV alpha grid size', choices=[500])
    parser.add_argument("--force", action="store_true", help="Replace existing evaluation outputs.")
    return parser.parse_args()


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
    model_tag = safe_tag(Path(args.model).name)
    strategy_tag = "-".join(strategies)
    return (
        f"model-{model_tag}__{strategy_tag}__{args.pad_attention_mode}__maskedpool__noBOSEOS"
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


def _mean_except_batch_and_feature(tensor: torch.Tensor) -> torch.Tensor:
    reduce_dims = tuple(range(1, tensor.ndim - 1))
    if not reduce_dims:
        return tensor.detach()
    return tensor.detach().mean(dim=reduce_dims)


def _masked_mean_except_batch_and_feature(tensor: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return _mean_except_batch_and_feature(tensor)
    if mask.shape != tensor.shape[:-1]:
        raise ValueError(f"Mask shape {tuple(mask.shape)} does not match tensor prefix {tuple(tensor.shape[:-1])}.")
    weights = mask.to(dtype=tensor.dtype).unsqueeze(-1)
    reduce_dims = tuple(range(1, tensor.ndim - 1))
    denom = weights.sum(dim=reduce_dims).clamp_min(1.0)
    return (tensor.detach() * weights).sum(dim=reduce_dims) / denom


def build_stage_pool_masks(seq_len: int, max_seq_len: tuple[int, ...], device: str | torch.device) -> list[torch.Tensor]:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive for masked pooling")
    multiple_of = int(np.prod(max_seq_len[1:]))
    pad_needed = (multiple_of - (seq_len % multiple_of)) % multiple_of
    padded_len = seq_len + pad_needed
    first_axis = padded_len // multiple_of

    valid_flat = torch.zeros((1, padded_len), dtype=torch.bool, device=device)
    valid_flat[:, :seq_len] = True
    valid_grid = valid_flat.view(1, first_axis, *max_seq_len[1:])

    stage_masks: list[torch.Tensor] = []
    n_axes = len(max_seq_len)
    for stage_idx in range(n_axes):
        if stage_idx < n_axes - 1:
            dims_to_reduce = tuple(range(1 + stage_idx + 1, 1 + n_axes))
            content_mask = valid_grid.any(dim=dims_to_reduce)
        else:
            content_mask = valid_grid
        start_mask = content_mask.any(dim=-1, keepdim=True)
        stage_masks.append(torch.cat([start_mask, content_mask], dim=-1))
    return stage_masks


def pool_hidden_states(
    hidden_states: list[torch.Tensor],
    strategy: str,
    stage_masks: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    if strategy != "all_concat_mean":
        raise ValueError("The paper protocol uses all_concat_mean pooling.")
    if stage_masks is not None and len(stage_masks) != len(hidden_states):
        raise ValueError(f"Expected {len(hidden_states)} stage masks, got {len(stage_masks)}")
    return torch.cat(
        [
            _masked_mean_except_batch_and_feature(hidden, None if stage_masks is None else stage_masks[index])
            for index, hidden in enumerate(hidden_states)
        ],
        dim=-1,
    )


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


def save_cache(cache_dir: Path, strategies: list[str], x_arrays: dict[str, np.ndarray], y_array: np.ndarray, sequences: np.ndarray) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    y_path, x_paths = cache_paths(cache_dir, strategies)
    np.save(y_path, y_array)
    np.save(cache_dir / "sequences.npy", sequences)
    for strategy in strategies:
        np.save(x_paths[strategy], x_arrays[strategy])


def load_cache(cache_dir: Path, strategies: list[str], df: pd.DataFrame, target_col: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    y_path, x_paths = cache_paths(cache_dir, strategies)
    y_array = np.load(y_path, allow_pickle=False)
    sequence_path = cache_dir / "sequences.npy"
    if not sequence_path.is_file():
        raise ValueError("Cache has no sequences.npy; regenerate the features with encode_only or full mode.")
    sequences = np.load(sequence_path, allow_pickle=False)
    expected_sequences = df["seq"].to_numpy(dtype=str)
    expected_targets = df[target_col].to_numpy(dtype=np.float32)
    if not np.array_equal(sequences, expected_sequences):
        raise ValueError("Cached sequences do not match the current input order; use a separate feature cache.")
    if y_array.dtype != np.float32 or not np.array_equal(y_array, expected_targets):
        raise ValueError("Cached targets do not match the current float32 fitness values.")
    x_arrays = {strategy: np.load(x_paths[strategy], allow_pickle=False) for strategy in strategies}
    return x_arrays, y_array


def load_model_bundle(args: argparse.Namespace) -> dict[str, object]:
    from phagen import load_model, load_tokenizer

    print("Loading PhaGen model...")
    model = load_model(
        args.model,
        device=args.device,
        torch_dtype=getattr(torch, args.dtype),
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    tokenizer = load_tokenizer(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    model.eval()
    runtime_device = str(next(model.parameters()).device)
    multiple_of = int(np.prod(model.config.max_seq_len[1:]))
    max_seq_len = tuple(int(x) for x in model.config.max_seq_len)
    print(f"Model ready: {args.model}, device={runtime_device}, dtype={args.dtype}")
    return {
        "model": model,
        "tokenizer": tokenizer,
        "runtime_device": runtime_device,
        "multiple_of": multiple_of,
        "max_seq_len": max_seq_len,
    }


def extract_features(
    bundle: dict[str, object],
    args: argparse.Namespace,
    strategies: list[str],
    df: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    runtime_device = bundle["runtime_device"]
    multiple_of = bundle["multiple_of"]
    max_seq_len = bundle["max_seq_len"]

    features = {strategy: [] for strategy in strategies}
    targets: list[float] = []
    total = len(df)
    progress_every = max(1, args.progress_every)

    for idx, row in enumerate(df.itertuples(index=False), start=1):
        seq = getattr(row, "seq")
        input_ids = tokenizer.encode(seq, add_special_tokens=False)
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=runtime_device).unsqueeze(0)

        with torch.no_grad():
            seq_len = input_tensor.shape[-1]
            pad_needed = (multiple_of - (seq_len % multiple_of)) % multiple_of
            attn_len = seq_len + pad_needed
            attention_mask = torch.ones((1, attn_len), dtype=torch.long, device=runtime_device)
            hidden_states = model(input_ids=input_tensor, attention_mask=attention_mask, is_causal=False).hidden_states

        stage_masks = build_stage_pool_masks(
            seq_len=input_tensor.shape[-1],
            max_seq_len=max_seq_len,
            device=runtime_device,
        )

        for strategy in strategies:
            vec = pool_hidden_states(hidden_states, strategy, stage_masks=stage_masks).squeeze(0).detach().to(dtype=torch.float32).cpu()
            features[strategy].append(vec)
        targets.append(float(getattr(row, args.target_col)))

        if idx % progress_every == 0 or idx == total:
            print(f"Embedded {idx}/{total}")

    x_arrays = {strategy: torch.stack(features[strategy], dim=0).numpy() for strategy in strategies}
    y_array = np.asarray(targets, dtype=np.float32)
    return x_arrays, y_array


def main() -> None:
    args = parse_args()
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
        targets = [output_json, output_csv]
        targets.extend(predictions_dir / f"oof_predictions__{strategy}.tsv" for strategy in strategies)
        existing = next((target for target in targets if target.exists()), None)
        if existing is not None:
            raise FileExistsError(f"Evaluation output already exists: {existing}; pass --force to replace it.")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run dir: {run_dir}")
    print(f"Model: {args.model}")
    print(f"Mutants path: {args.mutants_path}")
    print(f"Strategies: {strategies}")
    print(f"Total samples: {len(df)}")
    print(f"Sampling: mode={args.sample_mode}, seed={args.sample_seed}, max_samples={args.max_samples}")
    print(f"Pad attention mode: {args.pad_attention_mode}")
    print(f"Pool pad mode: {args.pool_pad_mode}")
    print(f"Add BOS/EOS: {bool(args.add_bos_eos)}")
    print(f"Use StandardScaler: {bool(args.use_standard_scaler)}")
    print(f"Run mode: {args.run_mode}")

    bundle_holder: dict[str, dict[str, object]] = {}

    def get_bundle() -> dict[str, object]:
        if "bundle" not in bundle_holder:
            bundle_holder["bundle"] = load_model_bundle(args)
        return bundle_holder["bundle"]

    run_encode = args.run_mode in {"full", "encode_only"}
    run_ridge = args.run_mode in {"full", "ridge_only"}
    cache_used = False
    cache_action = "none"
    x_arrays: dict[str, np.ndarray] = {}
    y_array: np.ndarray | None = None

    should_try_cache = args.run_mode == "ridge_only" or args.reuse_cached_features
    if should_try_cache and can_reuse_cache(cache_dir, strategies):
        x_arrays, y_array = load_cache(cache_dir, strategies, df, args.target_col)
        if all(x_arrays[strategy].shape[0] == len(df) for strategy in strategies) and y_array.shape[0] == len(df):
            cache_used = True
            cache_action = "reused"
            print(f"Reused cached features from: {cache_dir}")
        else:
            print("Cached feature shapes do not match current samples, recomputing.")
            x_arrays = {}
            y_array = None

    if y_array is None or not x_arrays:
        if run_encode:
            bundle = get_bundle()
            x_arrays, y_array = extract_features(bundle, args, strategies, df)
            save_cache(cache_dir, strategies, x_arrays, y_array, df["seq"].to_numpy(dtype=str))
            cache_action = "saved"
            print(f"Saved feature cache to: {cache_dir}")
        else:
            raise FileNotFoundError(f"Missing valid cached features for ridge_only: {cache_dir}")

    cache_status = {
        "cache_dir": str(cache_dir),
        "cache_action": cache_action,
        "cache_used": cache_used,
        "strategy_shapes": {strategy: list(x_arrays[strategy].shape) for strategy in strategies},
        "y_shape": list(y_array.shape),
    }

    if args.run_mode == "encode_only":
        print(f"Feature cache: {cache_dir}")
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

            pred_df = df.loc[:, [col for col in ["ambler_position", "pos", "mutation", args.target_col] if col in df.columns]].copy()
            pred_df[f"pred_{args.target_col}_oof"] = y_pred_oof
            pred_df.to_csv(predictions_dir / f"oof_predictions__{strategy}.tsv", sep="\t", index=False)

            rows.append(
                {
                    "strategy": strategy,
                    "n_samples": int(len(df)),
                    "model": args.model,
                    "pad_attention_mode": args.pad_attention_mode,
                    "pool_pad_mode": args.pool_pad_mode,
                    "add_bos_eos": bool(args.add_bos_eos),
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
        "model": args.model,
        "revision": args.revision,
        "local_files_only": args.local_files_only,
        "device": args.device,
        "dtype": args.dtype,
        "pad_attention_mode": args.pad_attention_mode,
        "pool_pad_mode": args.pool_pad_mode,
        "mutants_path": str(args.mutants_path),
        "results_dir": str(args.results_dir),
        "run_dir": str(run_dir),
        "run_mode": args.run_mode,
        "target_col": args.target_col,
        "strategies": strategies,
        "max_samples": args.max_samples,
        "sample_mode": args.sample_mode,
        "sample_seed": args.sample_seed,
        "add_bos_eos": bool(args.add_bos_eos),
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
    print(f"Model:         {args.model}")
    if ranking:
        for idx, (strategy, metrics) in enumerate(ranking, start=1):
            print(f"{idx}. {strategy}: mean_spearman={metrics['mean_spearman']:.4f}")
    print(f"Saved CSV:     {output_csv}")
    print(f"Saved JSON:    {output_json}")


if __name__ == "__main__":
    main()
