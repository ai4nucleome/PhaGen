#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import parser_for, evaluate


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
    print(
        f"Model ready: {args.model_name}, layer={args.primary_layer}, device={args.device}"
    )
    model.model.eval()
    return model


def pool_mean(embedding: torch.Tensor) -> np.ndarray:
    return embedding[0].mean(dim=0).detach().float().cpu().numpy()


POOL_FN = {"mean": pool_mean}


def run_forward(model, seq: str, primary_layer: str, device: str) -> torch.Tensor:
    input_ids = torch.tensor(
        model.tokenizer.tokenize(seq), dtype=torch.long, device=device
    ).unsqueeze(0)
    with torch.inference_mode():
        _, embeddings = model(
            input_ids,
            return_embeddings=True,
            layer_names=[primary_layer],
        )
    return embeddings[primary_layer]


def extract_features(
    model,
    args: argparse.Namespace,
    strategies: list[str],
    df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    features = {strategy: [] for strategy in strategies}
    total = len(df)
    progress_every = max(1, int(args.progress_every))
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        seq = str(getattr(row, "seq")).upper()
        embedding = run_forward(model, seq, args.primary_layer, args.device)
        for strategy in strategies:
            features[strategy].append(POOL_FN[strategy](embedding))
        if idx % progress_every == 0 or idx == total:
            print(f"Embedded {idx}/{total}")
    return {strategy: np.stack(values, axis=0) for strategy, values in features.items()}


def main():
    parser = parser_for("Evo2")
    parser.add_argument(
        "--model_name", choices=["evo2_1b_base", "evo2_7b"], default="evo2_7b"
    )
    parser.add_argument(
        "--model_path", type=Path, default=None, help="Optional local Evo2 checkpoint."
    )
    args = parser.parse_args()
    args.primary_layer = (
        "blocks.15" if args.model_name == "evo2_1b_base" else "blocks.26"
    )
    model_key = "evo2_1b" if args.model_name == "evo2_1b_base" else "evo2"
    return evaluate(
        args,
        model_key,
        "mean",
        lambda args, df: extract_features(load_model(args), args, ["mean"], df)["mean"],
    )


if __name__ == "__main__":
    raise SystemExit(main())
