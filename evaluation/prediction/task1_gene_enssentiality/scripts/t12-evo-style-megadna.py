#!/usr/bin/env python3


from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from t12_evo_style_utils import (
    build_wt_and_mut_inputs,
    build_output_path,
    ensure_output_available,
    gene_result,
    get_insert_cassette_pairs,
    load_inputs,
    make_parser,
    result_payload,
    save_evo_style_results,
    validate_paper_protocol,
)


_NT_VOCAB = ["**", "A", "T", "C", "G", "#"]


def parse_args():
    parser = make_parser('Native megaDNA sequence scores for zero-shot phage gene essentiality.')
    parser.add_argument(
        "--model-path", type=Path, default=os.environ.get("MEGADNA_MODEL_PATH")
    )
    parser.add_argument(
        "--model-repo", type=Path, default=os.environ.get("MEGADNA_REPO")
    )
    return parser.parse_args()


def encode_sequence(seq: str) -> list[int]:
    return [_NT_VOCAB.index(base) if base in _NT_VOCAB else 1 for base in seq]


def get_score(model, token_ids: list[int], device) -> float:
    inputs = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        loss = model(inputs, return_value="loss")
    value = loss.item() if isinstance(loss, torch.Tensor) else loss
    return -value


def run_one_context(
    model,
    genes: list[dict],
    genome_seq: str,
    context_bp: int = 8192,
    device: str = "cuda:0",
) -> list[dict]:
    validate_paper_protocol(context_bp)
    results = []
    for gene in tqdm(genes, desc="megaDNA/8192"):
        inputs = build_wt_and_mut_inputs(
            genome_seq, gene["start"], gene["end"], gene["strand"], context_bp
        )
        wt_tokens = encode_sequence(inputs["wt_seq"])
        mut_tokens = encode_sequence(inputs["mut_seq"])
        wt_score = get_score(model, wt_tokens, device)
        mut_score = get_score(model, mut_tokens, device)
        results.append(
            gene_result(
                gene,
                inputs,
                wt_score,
                mut_score,
                wt_ntok=len(wt_tokens),
                mut_ntok=len(mut_tokens),
                wt_add_bos_eos=False,
                mut_add_bos_eos=False,
            )
        )
    return results


def main():
    args = parse_args()
    output_path = build_output_path(args.results_dir, args.phage, "megadna")
    ensure_output_available(output_path, force=args.force)
    genome, genes = load_inputs(args)
    if args.model_path is None or not args.model_path.expanduser().is_file():
        raise FileNotFoundError(
            "Set --model-path or MEGADNA_MODEL_PATH to the local megaDNA checkpoint."
        )
    if args.model_repo is not None:
        repository = args.model_repo.expanduser().resolve()
        if not repository.is_dir():
            raise FileNotFoundError(repository)
        sys.path.insert(0, str(repository))
    model = torch.load(
        args.model_path.expanduser(),
        map_location=torch.device(args.device),
        weights_only=False,
    ).eval()
    results = run_one_context(model, genes, genome, device=args.device)
    payload = result_payload(
        args, "megadna", results, "mean_log_likelihood", bos_eos_mode="none"
    )
    save_evo_style_results(
        payload,
        output_path,
        force=args.force,
    )


if __name__ == "__main__":
    main()
