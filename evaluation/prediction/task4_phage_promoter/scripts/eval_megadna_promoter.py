#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import parser_for, evaluate

NT_VOCAB = ["**", "A", "T", "C", "G", "#"]
STRATEGY = "all_concat_mean"


def encode_sequence(seq: str, add_bos_eos: bool) -> list[int]:
    tokens = [NT_VOCAB.index(ch) if ch in NT_VOCAB else 1 for ch in seq]
    if add_bos_eos:
        tokens = [0] + tokens + [5]
    return tokens


def _mean_except_feature(tensor: torch.Tensor) -> torch.Tensor:
    reduce_dims = tuple(range(0, tensor.ndim - 1))
    if not reduce_dims:
        return tensor.detach()
    return tensor.detach().mean(dim=reduce_dims)


def pool_hidden_states(hidden_states: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([_mean_except_feature(hidden) for hidden in hidden_states], dim=-1)


def extract_features(
    model, args: argparse.Namespace, df: pd.DataFrame
) -> dict[str, np.ndarray]:
    rows = []
    total = len(df)
    progress_every = max(1, int(args.progress_every))
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        seq = str(getattr(row, "seq")).upper()
        token_ids = encode_sequence(seq, add_bos_eos=bool(args.add_bos_eos))
        input_tensor = torch.tensor(
            token_ids, dtype=torch.long, device=args.device
        ).unsqueeze(0)
        with torch.inference_mode():
            hidden_states = model(input_tensor, return_value="embedding")
        vec = (
            pool_hidden_states(hidden_states)
            .detach()
            .to(dtype=torch.float32)
            .cpu()
            .numpy()
        )
        rows.append(vec)
        if idx % progress_every == 0 or idx == total:
            print(f"Embedded {idx}/{total}")
    return {STRATEGY: np.stack(rows, axis=0)}


def load_model(args):
    if args.model_repo is not None:
        if not args.model_repo.is_dir():
            raise FileNotFoundError(args.model_repo)
        sys.path.insert(0, str(args.model_repo.resolve()))
    if not args.model_path.is_file():
        raise FileNotFoundError(args.model_path)
    model = torch.load(
        args.model_path, map_location=torch.device(args.device), weights_only=False
    )
    model.eval()
    return model


def main():
    parser = parser_for("megaDNA")
    parser.add_argument("--model_path", type=Path, required=True)
    args = parser.parse_args()
    args.add_bos_eos = 0
    return evaluate(
        args,
        "megadna",
        STRATEGY,
        lambda args, df: extract_features(load_model(args), args, df)[STRATEGY],
    )


if __name__ == "__main__":
    raise SystemExit(main())
