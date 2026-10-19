#!/usr/bin/env python3
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import torch
from common import parser_for, evaluate, NT_MODELS


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


def masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
    mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    pooled = (hidden * mask).sum(dim=1) / denom
    return pooled.detach().float().cpu().numpy()


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
    bundle: dict[str, object], args: argparse.Namespace, df: pd.DataFrame
) -> dict[str, np.ndarray]:
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    seqs = df["seq"].astype(str).str.upper().tolist()
    total = len(seqs)
    batch_size = max(1, int(args.batch_size))
    progress_every = max(1, int(args.progress_every))
    features: list[np.ndarray] = []

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = tokenizer(
            seqs[start:end],
            add_special_tokens=bool(args.add_special_tokens),
            padding=True,
            pad_to_multiple_of=args.pad_to_multiple_of,
            return_tensors="pt",
        )
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = build_attention_mask(batch, tokenizer, input_ids)
        pool_mask = pooling_attention_mask(
            attention_mask, tokenizer, input_ids, args.pool_special_tokens
        )
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        features.append(masked_mean(outputs.hidden_states[-1], pool_mask))
        if end % progress_every == 0 or end == total:
            print(f"Embedded {end}/{total}")
    return {"last_hidden_mean": np.concatenate(features, axis=0)}


def load_model_bundle(args, repo_id):
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    local_files_only = not args.allow_download
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id, trust_remote_code=True, local_files_only=local_files_only
    )
    model = AutoModelForMaskedLM.from_pretrained(
        repo_id,
        trust_remote_code=True,
        local_files_only=local_files_only,
        torch_dtype=torch.float32,
    )
    model.to(args.device)
    model.eval()
    return {"tokenizer": tokenizer, "model": model}


def main():
    parser = parser_for("nucleotide transformers")
    parser.add_argument("--model_name", choices=list(NT_MODELS), default="ntv3_8m")
    parser.add_argument(
        "--model", default=None, help="Optional local checkpoint directory."
    )
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()
    batch_sizes = {
        "ntv2_50m": 16,
        "ntv2_100m": 16,
        "ntv2_250m": 8,
        "ntv2_500m": 4,
        "ntv3_8m": 64,
        "ntv3_100m": 32,
        "ntv3_650m": 16,
    }
    args.batch_size = args.batch_size or batch_sizes[args.model_name]
    args.add_special_tokens = 0
    args.pad_to_multiple_of = 128
    args.pool_special_tokens = (
        "exclude" if args.model_name.startswith("ntv2_") else "include"
    )
    return evaluate(
        args,
        args.model_name,
        "last_hidden_mean",
        lambda args, df: extract_features(
            load_model_bundle(args, args.model or NT_MODELS[args.model_name]), args, df
        )["last_hidden_mean"],
    )


if __name__ == "__main__":
    raise SystemExit(main())
