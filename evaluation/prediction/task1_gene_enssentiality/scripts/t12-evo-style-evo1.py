#!/usr/bin/env python3


from __future__ import annotations

import os
import sys
from pathlib import Path

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


def parse_args():
    parser = make_parser('Native Evo1-7B sequence scores for zero-shot phage gene essentiality.')
    parser.add_argument(
        "--model-name", choices=["evo-1-131k-base"], default="evo-1-131k-base"
    )
    parser.add_argument("--model-repo", type=Path, default=os.environ.get("EVO_REPO"))
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def run_one_context(
    model,
    tokenizer,
    genes: list[dict],
    genome_seq: str,
    context_bp: int = 8192,
    device: str = "cuda:0",
) -> list[dict]:
    from evo import score_sequences

    validate_paper_protocol(context_bp)
    results = []
    for gene in tqdm(genes, desc="Evo1/8192"):
        inputs = build_wt_and_mut_inputs(
            genome_seq, gene["start"], gene["end"], gene["strand"], context_bp
        )
        wt_score = float(
            score_sequences([inputs["wt_seq"]], model, tokenizer, device=device)[0]
        )
        mut_score = float(
            score_sequences([inputs["mut_seq"]], model, tokenizer, device=device)[0]
        )
        results.append(gene_result(gene, inputs, wt_score, mut_score))
    return results


def main():
    args = parse_args()
    output_path = build_output_path(args.results_dir, args.phage, args.model_name)
    ensure_output_available(output_path, force=args.force)
    genome, genes = load_inputs(args)
    if args.model_repo is not None:
        repository = args.model_repo.expanduser().resolve()
        if not repository.is_dir():
            raise FileNotFoundError(repository)
        sys.path.insert(0, str(repository))
    if not args.allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
    from evo import Evo

    wrapper = Evo(args.model_name)
    model, tokenizer = wrapper.model, wrapper.tokenizer
    model.to(args.device).eval()
    results = run_one_context(model, tokenizer, genes, genome, device=args.device)
    payload = result_payload(args, args.model_name, results, "mean_log_likelihood")
    save_evo_style_results(
        payload,
        output_path,
        force=args.force,
    )


if __name__ == "__main__":
    main()
