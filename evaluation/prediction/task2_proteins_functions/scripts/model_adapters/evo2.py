

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch


def pool_mean(embeddings: dict, primary_layer: str, **_kw) -> np.ndarray:
    emb = embeddings[primary_layer]
    return emb[0].mean(dim=0).cpu().float().numpy()


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
    return model


def _run_forward(model, seq: str, layer_list: list[str], device: str) -> dict:
    input_ids = (
        torch.tensor(
            model.tokenizer.tokenize(seq),
            dtype=torch.int,
        )
        .unsqueeze(0)
        .to(device)
    )
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
        embeddings = _run_forward(
            model, getattr(row, "seq").upper(), [args.primary_layer], args.device
        )
        for strategy in strategies:
            features[strategy].append(pool_mean(embeddings, args.primary_layer))
        targets.append(float(getattr(row, args.target_col)))
        if index % progress_every == 0 or index == total:
            print(f"Embedded {index}/{total}")
    return (
        {strategy: np.stack(features[strategy], axis=0) for strategy in strategies},
        np.asarray(targets, dtype=np.float32),
    )
