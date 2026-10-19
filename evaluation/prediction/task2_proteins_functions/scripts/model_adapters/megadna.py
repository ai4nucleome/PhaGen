

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

NT_VOCAB = ["**", "A", "T", "C", "G", "#"]


def encode_sequence(seq: str) -> list[int]:
    return [NT_VOCAB.index(base) if base in NT_VOCAB else 1 for base in seq]


def _mean_except_feature(tensor: torch.Tensor) -> torch.Tensor:
    reduce_dims = tuple(range(0, tensor.ndim - 1))
    if not reduce_dims:
        return tensor.detach()
    return tensor.detach().mean(dim=reduce_dims)


def pool_hidden_states(
    hidden_states: list[torch.Tensor], strategy: str
) -> torch.Tensor:
    if strategy != "all_concat_mean":
        raise ValueError("The paper protocol uses all_concat_mean pooling.")
    return torch.cat([_mean_except_feature(hidden) for hidden in hidden_states], dim=-1)


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.model_repo is not None:
        model_repo = args.model_repo.expanduser().resolve()
        if not model_repo.is_dir():
            raise FileNotFoundError(f"Baseline repository not found: {model_repo}")
        sys.path.insert(0, str(model_repo))
    if not args.model_path:
        raise ValueError("Set --checkpoint to the megaDNA .pt checkpoint")
    if not Path(args.model_path).expanduser().is_file():
        raise FileNotFoundError(f"megaDNA checkpoint not found: {args.model_path}")
    print("Loading megaDNA model...")
    model = torch.load(
        Path(args.model_path).expanduser(),
        map_location=torch.device(args.device),
        weights_only=False,
    ).eval()

    model = model.to(device=args.device)
    print(f"Model ready: device={args.device}")
    return model


def extract_features(
    model: torch.nn.Module,
    args: argparse.Namespace,
    strategies: list[str],
    df: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    features = {strategy: [] for strategy in strategies}
    targets: list[float] = []
    total = len(df)
    progress_every = max(1, args.progress_every)

    for idx, row in enumerate(df.itertuples(index=False), start=1):
        input_ids = encode_sequence(getattr(row, "seq"))
        input_tensor = torch.tensor(
            input_ids, dtype=torch.long, device=args.device
        ).unsqueeze(0)

        with torch.inference_mode():
            hidden_states = model(input_tensor, return_value="embedding")

        for strategy in strategies:
            vec = (
                pool_hidden_states(hidden_states, strategy)
                .detach()
                .to(dtype=torch.float32)
                .cpu()
            )
            features[strategy].append(vec)
        targets.append(float(getattr(row, args.target_col)))

        if idx % progress_every == 0 or idx == total:
            print(f"Embedded {idx}/{total}")

    x_arrays = {
        strategy: torch.stack(features[strategy], dim=0).numpy()
        for strategy in strategies
    }
    y_array = np.asarray(targets, dtype=np.float32)
    return x_arrays, y_array
