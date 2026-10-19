#!/usr/bin/env python3


from __future__ import annotations

import os
import re
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
    parser = make_parser('Native Evo2 sequence scores for zero-shot phage gene essentiality.')
    parser.add_argument(
        "--model-name", choices=["evo2_1b_base", "evo2_7b"], default="evo2_7b"
    )
    parser.add_argument(
        "--model-path", type=Path, default=os.environ.get("EVO2_MODEL_PATH")
    )
    parser.add_argument("--model-repo", type=Path, default=os.environ.get("EVO2_REPO"))
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def run_one_context(
    model, genes: list[dict], genome_seq: str, context_bp: int = 8192
) -> list[dict]:
    validate_paper_protocol(context_bp)
    results = []
    for gene in tqdm(genes, desc="Evo2/8192"):
        inputs = build_wt_and_mut_inputs(
            genome_seq, gene["start"], gene["end"], gene["strand"], context_bp
        )
        wt_score = float(model.score_sequences([inputs["wt_seq"]])[0])
        mut_score = float(model.score_sequences([inputs["mut_seq"]])[0])
        results.append(gene_result(gene, inputs, wt_score, mut_score))
    return results


def configure_device(device: str) -> None:
    match = re.fullmatch(r"cuda:(\d+)", device)
    if match is None:
        raise ValueError("Evo2 requires --device cuda:<index>.")
    index = int(match.group(1))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        selected = str(index)
    else:
        devices = visible.split(",")
        if index >= len(devices) or not devices[index].strip():
            raise ValueError(
                "The requested Evo2 device is outside CUDA_VISIBLE_DEVICES."
            )
        selected = devices[index].strip()
    os.environ["CUDA_VISIBLE_DEVICES"] = selected


def main():
    args = parse_args()
    output_path = build_output_path(args.results_dir, args.phage, args.model_name)
    ensure_output_available(output_path, force=args.force)
    genome, genes = load_inputs(args)
    configure_device(args.device)
    if args.model_repo is not None:
        repository = args.model_repo.expanduser().resolve()
        if not repository.is_dir():
            raise FileNotFoundError(repository)
        sys.path.insert(0, str(repository))
    if not args.allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
    from evo2 import Evo2

    if args.model_path is not None:
        checkpoint = args.model_path.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model = Evo2(args.model_name, local_path=str(checkpoint))
    else:
        model = Evo2(args.model_name)
    model.model.eval()
    results = run_one_context(model, genes, genome)
    payload = result_payload(args, args.model_name, results, "mean_log_likelihood")
    save_evo_style_results(
        payload,
        output_path,
        force=args.force,
    )


if __name__ == "__main__":
    main()
