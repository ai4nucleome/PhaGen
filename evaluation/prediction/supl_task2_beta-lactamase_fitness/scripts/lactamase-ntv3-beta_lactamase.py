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
DEFAULT_RESULTS_DIR = TASK_DIR / "results" / "ntv3"
DEFAULT_MUTANTS_PATH = DATA_DIR / "mutants_sequences.tsv"
VALID_BASES = {"A", "C", "G", "T", "N"}
STRATEGY = "last_hidden_mean"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate nucleotide-transformer embeddings on TEM-1 beta-lactamase mutation fitness")
    parser.add_argument("--results_dir", type=Path, default=None, help="Model output directory")
    parser.add_argument("--model_name", default="InstaDeepAI/NTv3_100M_pre", help="Local model directory, Hugging Face repo id, or NTv3_* short name")
    parser.add_argument("--device", default="cuda:0", help='Inference device, e.g. "cuda:0"')
    parser.add_argument(
        "--torch_dtype",
        default="float32",
        choices=["auto", "float32", "bfloat16", "float16"],
        help="Model parameter dtype. float32 is safest for NTv3 remote code.",
    )
    parser.add_argument("--mutants_path", type=Path, default=DEFAULT_MUTANTS_PATH)
    parser.add_argument("--target_col", default="fitness")
    parser.add_argument('--outer_splits', type=int, default=5, choices=[5])
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all")
    parser.add_argument("--sample_mode", choices=["head", "random"], default="head")
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument('--use_standard_scaler', type=int, default=1, choices=[1], help='Standardize embeddings within each training fold.')
    parser.add_argument("--pad_to_multiple_of", type=int, choices=[128], default=128, help="Input padding multiple used in the paper.")
    parser.add_argument("--add_special_tokens", type=int, choices=[0, 1], default=0)
    parser.add_argument("--pool_special_tokens", choices=["include", "exclude"], default="include")
    parser.add_argument("--progress_every", type=int, default=500)
    parser.add_argument("--reuse_cached_features", action="store_true")
    parser.add_argument("--allow_download", action="store_true", help="Allow HuggingFace downloads. Default is local cache only.")
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
    parser.add_argument("--force", action="store_true", help="Replace existing evaluation outputs.")
    args = parser.parse_args()
    pair = (args.add_special_tokens, args.pool_special_tokens)
    model_name = args.model_name.lower()
    if pair not in {(0, "include"), (1, "exclude")}:
        parser.error("Use the paper token settings: NTv3=0/include or NTv2=1/exclude.")
    if "ntv3" in model_name and pair != (0, "include"):
        parser.error("NTv3 uses add_special_tokens=0 and pool_special_tokens=include.")
    if "nucleotide-transformer-v2" in model_name and pair != (1, "exclude"):
        parser.error("NTv2 uses add_special_tokens=1 and pool_special_tokens=exclude.")
    return args


def normalize_model_name(model_name: str) -> tuple[str, str]:
    local_path = Path(model_name).expanduser()
    repo_id = str(local_path.resolve()) if local_path.is_dir() else (model_name if "/" in model_name else f"InstaDeepAI/{model_name}")
    safe_name = repo_id.rsplit("/", 1)[-1]
    return repo_id, safe_name


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name in {"auto", "float32"}:
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    return torch.float32


def safe_tag(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw)


def sample_tag(max_samples: int, sample_mode: str, sample_seed: int) -> str:
    if max_samples <= 0:
        return "all"
    if sample_mode == "random":
        return f"random_{max_samples}_seed{sample_seed}"
    return f"head_{max_samples}"


def run_tag(args: argparse.Namespace, safe_model_name: str) -> str:
    special_tag = ""
    if bool(args.add_special_tokens) or args.pool_special_tokens != "include":
        add_tag = "addSpecial" if bool(args.add_special_tokens) else "noSpecial"
        special_tag = f"__{add_tag}__poolSpecial-{args.pool_special_tokens}"
    return (
        f"{safe_model_name}__{STRATEGY}{special_tag}"
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
    df["seq"] = df["seq"].astype(str).str.upper()
    bad_chars = sorted(set("".join(df["seq"].tolist())) - VALID_BASES)
    if bad_chars:
        raise ValueError(f"{mutants_path} contains unsupported DNA bases: {bad_chars}")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    if max_samples and max_samples > 0:
        n = min(max_samples, len(df))
        if sample_mode == "random":
            df = df.sample(n=n, random_state=sample_seed).reset_index(drop=True)
        else:
            df = df.head(n).copy()
    return df


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


def load_model_bundle(args: argparse.Namespace, repo_id: str) -> dict[str, object]:
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    local_files_only = not args.allow_download
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model = AutoModelForMaskedLM.from_pretrained(
        repo_id,
        trust_remote_code=True,
        local_files_only=local_files_only,
        torch_dtype=resolve_dtype(args.torch_dtype),
    )
    model.to(args.device)
    model.eval()
    print(f"Model ready: {repo_id}, device={args.device}, dtype={args.torch_dtype}")
    return {"tokenizer": tokenizer, "model": model}


def mean_pool_last_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
    mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    pooled = (hidden * mask).sum(dim=1) / denom
    return pooled.squeeze(0).detach().float().cpu().numpy()


def build_attention_mask(batch: dict[str, torch.Tensor], tokenizer, input_ids: torch.Tensor) -> torch.Tensor:
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        return attention_mask.to(input_ids.device)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return input_ids.ne(int(pad_token_id)).to(dtype=torch.long)
    return torch.ones_like(input_ids, dtype=torch.long)


def pooling_attention_mask(
    attention_mask: torch.Tensor,
    tokenizer,
    input_ids: torch.Tensor,
    pool_special_tokens: str,
) -> torch.Tensor:
    if pool_special_tokens == "include":
        return attention_mask
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        special_ids.discard(int(pad_token_id))
    if not special_ids:
        return attention_mask
    special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in special_ids:
        special_mask |= input_ids.eq(int(token_id))
    return attention_mask.masked_fill(special_mask, 0)


def extract_features(
    bundle: dict[str, object],
    args: argparse.Namespace,
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    features: list[np.ndarray] = []
    targets: list[float] = []
    total = len(df)
    progress_every = max(1, args.progress_every)

    for idx, row in enumerate(df.itertuples(index=False), start=1):
        seq = getattr(row, "seq")
        batch = tokenizer(
            [seq],
            add_special_tokens=bool(args.add_special_tokens),
            padding=True,
            pad_to_multiple_of=args.pad_to_multiple_of,
            return_tensors="pt",
        )
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = build_attention_mask(batch, tokenizer, input_ids)
        pool_mask = pooling_attention_mask(
            attention_mask,
            tokenizer,
            input_ids,
            args.pool_special_tokens,
        )

        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        features.append(mean_pool_last_hidden(outputs.hidden_states[-1], pool_mask))
        targets.append(float(getattr(row, args.target_col)))

        if idx % progress_every == 0 or idx == total:
            print(f"Embedded {idx}/{total}")

    return np.stack(features, axis=0), np.asarray(targets, dtype=np.float32)


def cache_paths(cache_dir: Path) -> tuple[Path, Path]:
    return cache_dir / "features.npy", cache_dir / "targets.npy"


def can_reuse_cache(cache_dir: Path) -> bool:
    x_path, y_path = cache_paths(cache_dir)
    return x_path.exists() and y_path.exists()


def save_cache(cache_dir: Path, x_array: np.ndarray, y_array: np.ndarray, df: pd.DataFrame) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    x_path, y_path = cache_paths(cache_dir)
    np.save(x_path, x_array)
    np.save(y_path, y_array)
    np.save(cache_dir / "sequences.npy", df["seq"].astype(str).to_numpy(dtype=str))


def cache_matches_data(cache_dir: Path, df: pd.DataFrame, target_col: str, y_array: np.ndarray) -> bool:
    sequence_path = cache_dir / "sequences.npy"
    if not sequence_path.is_file():
        return False
    cached_sequences = np.load(sequence_path, allow_pickle=False)
    expected_sequences = df["seq"].astype(str).to_numpy(dtype=str)
    expected_targets = df[target_col].to_numpy(dtype=np.float32)
    return np.array_equal(cached_sequences, expected_sequences) and np.array_equal(y_array, expected_targets)


def load_cache(cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    x_path, y_path = cache_paths(cache_dir)
    return np.load(x_path), np.load(y_path)


def main() -> None:
    args = parse_args()
    repo_id, safe_model_name = normalize_model_name(args.model_name)
    if args.results_dir is None:
        if safe_model_name.startswith("nucleotide-transformer-v2-"):
            size = safe_model_name[len("nucleotide-transformer-v2-"):].removesuffix("-multi-species")
            key = f"ntv2_{size}"
        else:
            key = safe_model_name.lower().removesuffix("_pre")
        args.results_dir = TASK_DIR / "results" / key
    args.results_dir = args.results_dir.resolve()
    args.mutants_path = args.mutants_path.resolve()

    if not args.mutants_path.is_file():
        raise FileNotFoundError(f"Mutants TSV not found: {args.mutants_path}")
    if args.run_mode != "encode_only" and args.outer_splits < 2:
        raise ValueError("--outer_splits must be at least 2")

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
        targets.extend(predictions_dir / f"oof_predictions__{strategy}.tsv" for strategy in [STRATEGY])
        existing = next((target for target in targets if target.exists()), None)
        if existing is not None:
            raise FileExistsError(f"Evaluation output already exists: {existing}; pass --force to replace it.")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run dir: {run_dir}")
    print(f"Mutants path: {args.mutants_path}")
    print(f"Model: {repo_id}")
    print(f"Pooling: {STRATEGY}")
    print(f"Total samples: {len(df)}")
    print(f"Sampling: mode={args.sample_mode}, seed={args.sample_seed}, max_samples={args.max_samples}")
    print(f"Use StandardScaler: {bool(args.use_standard_scaler)}")
    print(f"Add special tokens: {bool(args.add_special_tokens)}")
    print(f"Pool special tokens: {args.pool_special_tokens}")
    print(f"Run mode: {args.run_mode}")

    bundle_holder: dict[str, dict[str, object]] = {}

    def get_bundle() -> dict[str, object]:
        if "bundle" not in bundle_holder:
            bundle_holder["bundle"] = load_model_bundle(args, repo_id)
        return bundle_holder["bundle"]

    run_encode = args.run_mode in {"full", "encode_only"}
    run_ridge = args.run_mode in {"full", "ridge_only"}
    cache_used = False
    cache_action = "none"
    x_array: np.ndarray | None = None
    y_array: np.ndarray | None = None

    should_try_cache = args.run_mode == "ridge_only" or args.reuse_cached_features
    if should_try_cache and can_reuse_cache(cache_dir):
        x_array, y_array = load_cache(cache_dir)
        if x_array.ndim == 2 and x_array.shape[0] == len(df) and cache_matches_data(cache_dir, df, args.target_col, y_array):
            cache_used = True
            cache_action = "reused"
            print(f"Reused cached features: X={x_array.shape}, y={y_array.shape}")
        else:
            print("Cache sequence order, targets, or feature shapes do not match the input; cache rejected.")
            x_array = None
            y_array = None

    if x_array is None or y_array is None:
        if run_encode:
            x_array, y_array = extract_features(get_bundle(), args, df)
            save_cache(cache_dir, x_array, y_array, df)
            cache_action = "saved"
            print(f"Saved feature cache: X={x_array.shape}, y={y_array.shape}")
        else:
            raise FileNotFoundError(f"Missing aligned cache for ridge_only: {cache_dir}. Regenerate features for the current input; sequences.npy and targets.npy must match every input row.")

    cache_status = {
        "cache_dir": str(cache_dir),
        "cache_action": cache_action,
        "cache_used": cache_used,
        "x_shape": list(x_array.shape),
        "y_shape": list(y_array.shape),
    }

    rows: list[dict] = []
    if args.run_mode == "encode_only":
        print(f"Feature cache: {cache_dir}")
        return

    results: dict[str, dict] = {}
    if run_ridge:
        predictions_dir.mkdir(parents=True, exist_ok=True)
        metrics, y_pred_oof = evaluate_with_ridge(
            x_array=x_array,
            y_array=y_array,
            outer_splits=args.outer_splits,
            n_jobs=args.n_jobs,
            alpha_log_min=args.alpha_log_min,
            alpha_log_max=args.alpha_log_max,
            alpha_num=args.alpha_num,
            use_standard_scaler=bool(args.use_standard_scaler),
        )
        results[STRATEGY] = metrics

        pred_df = df.loc[:, [col for col in ["ambler_position", "pos", "mutation", args.target_col] if col in df.columns]].copy()
        pred_df[f"pred_{args.target_col}_oof"] = y_pred_oof
        pred_df.to_csv(predictions_dir / f"oof_predictions__{STRATEGY}.tsv", sep="\t", index=False)

        rows.append(
            {
                "strategy": STRATEGY,
                "n_samples": int(len(df)),
                "model_name": repo_id,
                "pooling_tag": STRATEGY,
                "add_special_tokens": bool(args.add_special_tokens),
                "pool_special_tokens": args.pool_special_tokens,
                "use_standard_scaler": bool(args.use_standard_scaler),
                "cache_used": cache_used,
                "mean_spearman": metrics["mean_spearman"],
                "std_spearman": metrics["std_spearman"],
                "spearman_folds": json.dumps(metrics["spearman_folds"]),
                "selected_alphas": json.dumps(metrics["selected_alphas"]),
            }
        )
        print(f"RESULT {STRATEGY}: mean_spearman={metrics['mean_spearman']:.4f}")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    payload = {
        "model_name": repo_id,
        "device": args.device,
        "torch_dtype": args.torch_dtype,
        "pooling_tag": STRATEGY,
        "mutants_path": str(args.mutants_path),
        "results_dir": str(args.results_dir),
        "run_dir": str(run_dir),
        "run_mode": args.run_mode,
        "target_col": args.target_col,
        "max_samples": args.max_samples,
        "sample_mode": args.sample_mode,
        "sample_seed": args.sample_seed,
        "pad_to_multiple_of": args.pad_to_multiple_of,
        "add_special_tokens": bool(args.add_special_tokens),
        "pool_special_tokens": args.pool_special_tokens,
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
        "output_csv": str(output_csv),
        "output_predictions_dir": str(predictions_dir),
    }
    output_json.write_text(json.dumps(payload, indent=2))

    print("\n=== Summary ===")
    print(f"Run mode:      {args.run_mode}")
    print(f"Cache action:  {cache_action}")
    if rows:
        print(f"Mean fold Spearman: {rows[0]['mean_spearman']:.4f}")
    print(f"Saved CSV:     {output_csv}")
    print(f"Saved JSON:    {output_json}")


if __name__ == "__main__":
    main()
