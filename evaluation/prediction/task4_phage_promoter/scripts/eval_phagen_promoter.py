#!/usr/bin/env python3
from __future__ import annotations
import argparse
import numpy as np
import torch
from common import parser_for, evaluate


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
    if tensor.shape[:-1] != mask.shape:
        raise ValueError(
            f"Pooling mask shape {tuple(mask.shape)} does not match tensor shape {tuple(tensor.shape)}"
        )
    reduce_dims = tuple(range(1, tensor.ndim - 1))
    if not reduce_dims:
        return tensor.detach()
    mask = mask.to(device=tensor.device, dtype=tensor.dtype)
    denom = mask.sum(dim=reduce_dims).clamp_min(1.0).unsqueeze(-1)
    return (tensor.detach() * mask.unsqueeze(-1)).sum(dim=reduce_dims) / denom


def build_stage_pool_masks(
    seq_len: int, max_seq_len: tuple[int, ...], device: str | torch.device
) -> list[torch.Tensor]:
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


def pool_hidden_states(hidden_states, strategy, stage_masks):
    if strategy != "all_concat_mean":
        raise ValueError("The paper uses all_concat_mean pooling.")
    if stage_masks is not None and len(stage_masks) != len(hidden_states):
        raise ValueError("Stage mask count does not match hidden states")
    return torch.cat(
        [
            _masked_mean_except_batch_and_feature(
                hidden, None if stage_masks is None else stage_masks[index]
            )
            for index, hidden in enumerate(hidden_states)
        ],
        dim=-1,
    )


def extract_features(bundle, args, strategies, df):
    model, tokenizer = bundle["model"], bundle["tokenizer"]
    device = bundle["runtime_device"]
    multiple_of = bundle["multiple_of"]
    features = []
    for index, row in enumerate(df.itertuples(index=False), start=1):
        token_ids = tokenizer.encode(str(row.seq).upper(), add_special_tokens=False)
        inputs = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
        padded_length = (
            inputs.shape[-1]
            + (multiple_of - inputs.shape[-1] % multiple_of) % multiple_of
        )
        attention = torch.ones((1, padded_length), dtype=torch.long, device=device)
        with torch.inference_mode():
            hidden = model(input_ids=inputs, attention_mask=attention).hidden_states
        masks = build_stage_pool_masks(inputs.shape[-1], bundle["max_seq_len"], device)
        vector = (
            pool_hidden_states(hidden, "all_concat_mean", masks)
            .squeeze(0)
            .detach()
            .to(dtype=torch.float32)
            .cpu()
            .numpy()
        )
        features.append(vector)
        if index % max(1, args.progress_every) == 0 or index == len(df):
            print(f"Embedded {index}/{len(df)}")
    return {"all_concat_mean": np.stack(features, axis=0)}


def main():
    parser = parser_for("PhaGen")
    parser.add_argument("--model", default="YusenWithR/PhaGen")
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()
    args.dtype = "float32"
    args.local_files_only = not args.allow_download
    args.add_bos_eos = 0
    args.pad_attention_mode = "attend_pad"
    args.pool_pad_mode = "exclude_pad"
    strategy = "all_concat_mean"

    def encode(args, df):
        return extract_features(load_model_bundle(args), args, [strategy], df)[strategy]

    return evaluate(args, "phagen", strategy, encode)


if __name__ == "__main__":
    raise SystemExit(main())
