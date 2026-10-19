

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch


def _mean_except_batch_and_feature(tensor: torch.Tensor) -> torch.Tensor:
    reduce_dims = tuple(range(1, tensor.ndim - 1))
    if not reduce_dims:
        return tensor.detach()
    return tensor.detach().mean(dim=reduce_dims)


def _masked_mean_except_batch_and_feature(
    tensor: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    if mask is None:
        return _mean_except_batch_and_feature(tensor)
    if mask.shape != tensor.shape[:-1]:
        raise ValueError(
            f"Mask shape {tuple(mask.shape)} does not match tensor prefix {tuple(tensor.shape[:-1])}."
        )
    weights = mask.to(dtype=tensor.dtype).unsqueeze(-1)
    reduce_dims = tuple(range(1, tensor.ndim - 1))
    denom = weights.sum(dim=reduce_dims).clamp_min(1.0)
    return (tensor.detach() * weights).sum(dim=reduce_dims) / denom


def build_stage_pool_masks(
    seq_len: int, max_seq_len: tuple[int, ...], device: str | torch.device
) -> list[torch.Tensor]:
    
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
        raise ValueError(
            f"Expected {len(hidden_states)} stage masks, got {len(stage_masks)}"
        )
    return torch.cat(
        [
            _masked_mean_except_batch_and_feature(
                hidden, None if stage_masks is None else stage_masks[index]
            )
            for index, hidden in enumerate(hidden_states)
        ],
        dim=-1,
    )


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
        input_tensor = torch.tensor(
            input_ids, dtype=torch.long, device=runtime_device
        ).unsqueeze(0)

        with torch.no_grad():
            seq_len = input_tensor.shape[-1]
            pad_needed = (multiple_of - (seq_len % multiple_of)) % multiple_of
            attn_len = seq_len + pad_needed
            attention_mask = torch.ones(
                (1, attn_len), dtype=torch.long, device=runtime_device
            )
            hidden_states = model(
                input_ids=input_tensor, attention_mask=attention_mask, is_causal=False
            ).hidden_states

        stage_masks = build_stage_pool_masks(
            seq_len=input_tensor.shape[-1],
            max_seq_len=max_seq_len,
            device=runtime_device,
        )

        for strategy in strategies:
            vec = (
                pool_hidden_states(hidden_states, strategy, stage_masks=stage_masks)
                .squeeze(0)
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
