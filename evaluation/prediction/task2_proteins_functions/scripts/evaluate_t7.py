#!/usr/bin/env python3


from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TASK_DIR = Path(__file__).resolve().parent.parent
MODELS = {
    "phagen": ("PhaGen", "phagen", "YusenWithR/PhaGen", "all_concat_mean", None),
    "megadna": ("megaDNA", "megadna", "megaDNA", "all_concat_mean", None),
    **{
        f"ntv2_{size}m": (
            f"NTv2-{size}M",
            "ntv3",
            f"InstaDeepAI/nucleotide-transformer-v2-{size}m-multi-species",
            "last_hidden_mean",
            None,
        )
        for size in [50, 100, 250, 500]
    },
    **{
        f"ntv3_{size}m": (
            f"NTv3-{size}M",
            "ntv3",
            f"InstaDeepAI/NTv3_{size}M_pre",
            "last_hidden_mean",
            None,
        )
        for size in [8, 100, 650]
    },
    "evo": ("Evo1-7B", "evo", "evo-1-131k-base", "mean", "blocks.3"),
    "evo2_1b": ("Evo2-1B", "evo2", "evo2_1b_base", "mean", "blocks.15"),
    "evo2": ("Evo2-7B", "evo2", "evo2_7b", "mean", "blocks.26"),
}


def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rho, _ = spearmanr(y_pred, y_true, nan_policy="omit")
    return 0.0 if np.isnan(rho) else float(rho)


def evaluate_gene(
    x: np.ndarray, y: np.ndarray, n_jobs: int = 1
) -> tuple[dict, np.ndarray]:
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    probe = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        RidgeCV(alphas=np.logspace(-6, 6, 500)),
    )
    cv = cross_validate(
        probe,
        x,
        y,
        cv=folds,
        scoring={"spearman": make_scorer(spearman_corr, greater_is_better=True)},
        n_jobs=n_jobs,
        return_estimator=True,
    )
    predictions = np.empty_like(y)
    for estimator, (_, test) in zip(cv["estimator"], folds.split(x, y)):
        predictions[test] = estimator.predict(x[test])
    return {
        "mean_spearman": float(np.mean(cv["test_spearman"])),
        "std_spearman": float(np.std(cv["test_spearman"])),
        "spearman_folds": [float(v) for v in cv["test_spearman"]],
        "selected_alphas": [
            float(est.named_steps["ridgecv"].alpha_) for est in cv["estimator"]
        ],
    }, predictions


def read_gene(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype={"gene": str})
    required = {"gene", "seq", "nmi", "mutation", "mut_type"}
    if not required.issubset(frame):
        raise ValueError(f"Missing columns in {path}: {sorted(required - set(frame))}")
    frame = frame.dropna(subset=["seq", "nmi"]).copy()
    frame["nmi"] = pd.to_numeric(frame["nmi"], errors="coerce")
    frame = frame.dropna(subset=["nmi"]).reset_index(drop=True)
    if len(frame) < 10 or not np.isfinite(frame["nmi"]).all():
        raise ValueError(f"Expected at least ten finite targets in {path}")
    return frame


def load_cached_gene(
    cache: Path, strategy: str, frame: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    feature = next(
        (
            p
            for p in [cache / f"X_{strategy}.npy", cache / "features.npy"]
            if p.is_file()
        ),
        None,
    )
    if feature is None:
        raise FileNotFoundError(f"No {strategy} feature matrix in {cache}")
    x = np.load(feature, mmap_mode="r", allow_pickle=False)
    y = np.load(cache / "targets.npy", allow_pickle=False)
    expected = frame["nmi"].to_numpy(dtype=np.float32)
    if (
        x.ndim != 2
        or x.shape[0] != len(frame)
        or y.shape != expected.shape
        or not np.array_equal(y, expected)
    ):
        raise ValueError(f"Feature-cache rows or targets do not match {cache}")
    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite features in {cache}")
    sequences = cache / "sequences.npy"
    if sequences.exists() and not np.array_equal(
        np.load(sequences, allow_pickle=False), frame["seq"].to_numpy(dtype=str)
    ):
        raise ValueError(f"Cached sequence order differs in {cache}")
    return np.asarray(x, dtype=np.float32), y


def save_cache(
    cache: Path, strategy: str, x: np.ndarray, y: np.ndarray, frame: pd.DataFrame
) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / f"X_{strategy}.npy", x)
    np.save(cache / "targets.npy", y)
    np.save(cache / "sequences.npy", frame["seq"].to_numpy(dtype=str))


def make_encoder(arguments: argparse.Namespace):
    _, family, model_id, strategy, layer = MODELS[arguments.model_key]
    adapter = importlib.import_module(f"model_adapters.{family}")
    settings = SimpleNamespace(
        model=arguments.checkpoint or model_id,
        model_name=arguments.checkpoint
        if family == "ntv3" and arguments.checkpoint
        else model_id,
        model_path=Path(arguments.checkpoint)
        if arguments.checkpoint and family in {"megadna", "evo2"}
        else None,
        model_repo=arguments.model_repo,
        device=arguments.device,
        dtype="float32",
        torch_dtype="float32",
        revision=arguments.revision,
        local_files_only=not arguments.allow_download,
        allow_download=arguments.allow_download,
        primary_layer=layer,
        target_col="nmi",
        progress_every=arguments.progress_every,
        add_special_tokens=0,
        pool_special_tokens="exclude"
        if arguments.model_key.startswith("ntv2_")
        else "include",
        pad_to_multiple_of=128,
        pad_attention_mode="attend_pad",
        pool_pad_mode="exclude_pad",
        add_bos_eos=0,
    )
    if family == "ntv3":
        bundle = adapter.load_model_bundle(settings, settings.model_name)

        def encode(frame):
            return adapter.extract_features(bundle, settings, frame)
    else:
        bundle = (
            adapter.load_model(settings)
            if family in {"megadna", "evo2"}
            else adapter.load_model_bundle(settings)
        )

        def encode(frame):
            features, targets = adapter.extract_features(
                bundle, settings, [strategy], frame
            )
            return features[strategy], targets

    return encode


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(default_model: str = "phagen") -> None:
    parser = argparse.ArgumentParser(description='Evaluate frozen representations with independent five-fold T7 gene probes.')
    parser.add_argument("--model-key", choices=list(MODELS), default=default_model)
    parser.add_argument("--data-dir", type=Path, default=TASK_DIR / "data/processed")
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        help="Model cache root containing gene_<name>/pooling_cache/.",
    )
    parser.add_argument(
        "--checkpoint",
        help="Local PhaGen/NT model directory, megaDNA checkpoint or Evo2 checkpoint.",
    )
    parser.add_argument(
        "--model-repo", type=Path, help="Optional local baseline source repository."
    )
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--genes",
        default="",
        help="Comma-separated gene identifiers; default evaluates all 59.",
    )
    parser.add_argument(
        "--run-mode", choices=["full", "encode_only", "ridge_only"], default="full"
    )
    parser.add_argument("--reuse-cached-features", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    label, _, model_id, strategy, layer = MODELS[args.model_key]
    results = args.results_dir or TASK_DIR / "results" / args.model_key
    cache_root = args.feature_cache_dir or TASK_DIR / "feature_cache" / args.model_key
    requested = {v.strip() for v in args.genes.split(",") if v.strip()}
    paths = [
        (p.stem.removeprefix("gene_"), p)
        for p in sorted(args.data_dir.glob("gene_*.tsv"))
    ]
    if requested:
        paths = [(gene, p) for gene, p in paths if gene in requested]
        missing = requested - {gene for gene, _ in paths}
        if missing:
            raise FileNotFoundError(f"Missing gene tables: {sorted(missing)}")
    if not paths:
        raise FileNotFoundError(f"No gene tables in {args.data_dir}")
    outputs = [
        results / "summary.json",
        results / "per_gene_results.csv",
        *[results / f"oof_predictions__gene_{gene}.tsv" for gene, _ in paths],
    ]
    if args.run_mode != "encode_only" and not args.force:
        existing = next((p for p in outputs if p.exists()), None)
        if existing:
            raise FileExistsError(
                f"Output already exists: {existing}; use --force to replace it."
            )
    encoder = None
    rows = []
    for gene, path in paths:
        frame = read_gene(path)
        cache = cache_root / f"gene_{gene}" / "pooling_cache"
        if args.run_mode == "ridge_only" or args.reuse_cached_features:
            x, y = load_cached_gene(cache, strategy, frame)
        else:
            if encoder is None:
                encoder = make_encoder(args)
            x, y = encoder(frame)
            save_cache(cache, strategy, x, y, frame)
        if args.run_mode == "encode_only":
            continue
        metrics, predictions = evaluate_gene(x, y, args.n_jobs)
        results.mkdir(parents=True, exist_ok=True)
        table = frame[["gene", "mutation", "mut_type", "nmi"]].copy()
        table["pred_nmi_oof"] = predictions
        table.to_csv(
            results / f"oof_predictions__gene_{gene}.tsv", sep="\t", index=False
        )
        rows.append(
            {
                "gene": gene,
                "strategy": strategy,
                "n_samples": len(frame),
                "source_tsv": f"data/processed/{path.name}",
                **{
                    k: json.dumps(v) if isinstance(v, list) else v
                    for k, v in metrics.items()
                },
            }
        )
        print(
            f"{gene}: n={len(frame)}, mean fold Spearman={metrics['mean_spearman']:.6f}"
        )
    if not rows:
        return
    write_rows(results / "per_gene_results.csv", rows)
    summary = {
        "model_key": args.model_key,
        "model": label,
        "model_name": model_id,
        "strategy": strategy,
        "n_genes": len(rows),
        "n_samples": sum(row["n_samples"] for row in rows),
        "use_standard_scaler": True,
        "outer_splits": 5,
        "seed": 42,
        "alpha_log_min": -6.0,
        "alpha_log_max": 6.0,
        "alpha_num": 500,
        "mean_spearman": float(np.mean([row["mean_spearman"] for row in rows])),
        "oof_predictions_available": True,
    }
    if layer:
        summary["primary_layer"] = layer
    if args.model_key == "phagen":
        summary.update(
            {
                "resolved_step": 12800,
                "pad_attention_mode": "attend_pad",
                "pool_pad_mode": "exclude_pad",
            }
        )
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
