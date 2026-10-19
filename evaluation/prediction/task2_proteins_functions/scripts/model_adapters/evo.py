

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch


def pool_mean(embedding: torch.Tensor) -> np.ndarray:
    return embedding[0].mean(dim=0).float().cpu().numpy()


POOL_FN = {"mean": pool_mean}


def load_model_bundle(args: argparse.Namespace) -> dict[str, object]:
    if args.model_repo is not None:
        model_repo = args.model_repo.expanduser().resolve()
        if not model_repo.is_dir():
            raise FileNotFoundError(f"Baseline repository not found: {model_repo}")
        sys.path.insert(0, str(model_repo))
    if not args.allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
    from evo import Evo

    class IdentityUnembed(torch.nn.Module):
        def unembed(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states

    print(f"Loading Evo model: {args.model_name}")
    evo_model = Evo(args.model_name)
    model, tokenizer = evo_model.model, evo_model.tokenizer
    model.to(args.device)
    model.unembed = IdentityUnembed().to(args.device)
    model.eval()

    capture: dict[str, torch.Tensor] = {}
    module_lookup = dict(model.named_modules())
    if args.primary_layer not in module_lookup:
        available = sorted(name for name in module_lookup if name)
        preview = available[:40]
        raise ValueError(
            f"Unknown primary layer: {args.primary_layer}. "
            f"Example available modules: {preview}"
        )
    target_module = module_lookup[args.primary_layer]

    def _capture_hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        capture["embedding"] = hidden.detach()

    hook_handle = target_module.register_forward_hook(_capture_hook)

    print(
        f"Model ready: {args.model_name}, layer={args.primary_layer}, device={args.device}"
    )
    bundle = {
        "evo_model": evo_model,
        "model": model,
        "tokenizer": tokenizer,
        "capture": capture,
        "hook_handle": hook_handle,
    }
    return bundle


def _run_forward(
    bundle: dict[str, object], seq: str, primary_layer: str, device: str
) -> torch.Tensor:
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    input_ids = torch.tensor(
        tokenizer.tokenize(seq),
        dtype=torch.int,
        device=device,
    ).unsqueeze(0)

    capture = bundle["capture"]
    capture.clear()
    with torch.no_grad():
        embedding, _ = model(input_ids)
    if "embedding" not in capture:
        raise RuntimeError(f"Failed to capture embedding from layer: {primary_layer}")
    return capture["embedding"]


def extract_features(
    bundle: dict[str, object],
    args: argparse.Namespace,
    strategies: list[str],
    df: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    features = {strategy: [] for strategy in strategies}
    targets: list[float] = []
    total = len(df)
    progress_every = max(1, args.progress_every)

    for idx, row in enumerate(df.itertuples(index=False), start=1):
        seq = getattr(row, "seq").upper()
        target = float(getattr(row, args.target_col))
        embedding = _run_forward(bundle, seq, args.primary_layer, args.device)
        for strategy in strategies:
            features[strategy].append(POOL_FN[strategy](embedding))
        targets.append(target)

        if idx % progress_every == 0 or idx == total:
            print(f"Embedded {idx}/{total}")

    x_arrays = {
        strategy: np.stack(features[strategy], axis=0) for strategy in strategies
    }
    y_array = np.asarray(targets, dtype=np.float32)
    return x_arrays, y_array
