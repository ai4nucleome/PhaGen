#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from embedding_utils.encoders import FrozenEncoder, MODEL_SPECS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=MODEL_SPECS, required=True)
    parser.add_argument("--data-path", type=Path, default=ROOT / "data/processed/phavip_threshold40.tsv")
    parser.add_argument("--sequence-column", default="dna_seq")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", help="PhaGen/NT checkpoint directory or model ID; Evo model name")
    parser.add_argument("--model-path", type=Path, help="Local MegaDNA or Evo2 checkpoint")
    parser.add_argument("--model-repo", type=Path, help="Local baseline package checkout")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--pool-special-tokens", choices=["include", "exclude"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    cache_dir = args.cache_dir or ROOT / "feature_cache" / args.model_key
    strategy = MODEL_SPECS[args.model_key][1]
    outputs = [cache_dir / f"all_X__{strategy}.npy", cache_dir / "record_ids.npy", cache_dir / "sequences.npy"]
    if any(path.exists() for path in outputs) and not args.force:
        raise FileExistsError(f"Cache output already exists in {cache_dir}; use --force")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    table = pd.read_csv(args.data_path, sep="\t", low_memory=False, keep_default_na=False)
    sequences = table[args.sequence_column].astype(str).str.upper().tolist()
    if not sequences or any(not seq or set(seq) - set("ACGT") for seq in sequences):
        raise ValueError("Expected nonempty DNA sequences containing only A/C/G/T")
    encoder = FrozenEncoder(args.model_key, args.device, args.model, args.model_path,
                            args.model_repo, args.allow_download, args.pool_special_tokens)
    arrays = []
    try:
        for start in range(0, len(sequences), args.batch_size):
            arrays.append(encoder.encode_batch(sequences[start:start + args.batch_size]))
            print(f"Encoded {min(start + args.batch_size, len(sequences))}/{len(sequences)}", flush=True)
    finally:
        encoder.close()
    features = np.concatenate(arrays, axis=0).astype(np.float32, copy=False)
    expected = (len(sequences), MODEL_SPECS[args.model_key][2])
    if features.shape != expected or not np.isfinite(features).all():
        raise ValueError(f"Invalid embedding array: {features.shape}; expected {expected}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(outputs[0], features)
    np.save(outputs[1], np.arange(len(sequences), dtype=np.int64))
    np.save(outputs[2], np.asarray(sequences))


if __name__ == "__main__":
    main()
