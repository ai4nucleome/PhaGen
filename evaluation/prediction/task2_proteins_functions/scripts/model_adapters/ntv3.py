

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name in {"auto", "float32"}:
        return torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    return torch.float32


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


def mean_pool_last_hidden(
    hidden: torch.Tensor, attention_mask: torch.Tensor
) -> np.ndarray:
    mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    pooled = (hidden * mask).sum(dim=1) / denom
    return pooled.squeeze(0).detach().float().cpu().numpy()


def build_attention_mask(
    batch: dict[str, torch.Tensor], tokenizer, input_ids: torch.Tensor
) -> torch.Tensor:
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
