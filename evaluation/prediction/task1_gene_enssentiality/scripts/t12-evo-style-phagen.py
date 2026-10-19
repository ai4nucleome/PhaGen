#!/usr/bin/env python3


from __future__ import annotations

import math

import torch
import torch.nn.functional as F
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
    parser = make_parser('PhaGen masked-region scores for zero-shot phage gene essentiality.')
    parser.add_argument(
        "--model",
        default="YusenWithR/PhaGen",
        help="Local paper checkpoint directory or model ID.",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def build_attention_mask(model, token_ids: list[int], device) -> torch.Tensor:
    multiple = int(math.prod(model.config.max_seq_len[1:]))
    padded_length = len(token_ids) + (-len(token_ids)) % multiple
    return torch.ones((1, padded_length), dtype=torch.long, device=device)


def get_masked_logprobs(
    model, token_ids: list[int], positions: list[int], device, mask_token_id: int
) -> list[float]:
    masked = list(token_ids)
    for position in positions:
        masked[position] = mask_token_id
    input_ids = torch.tensor(masked, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = build_attention_mask(model, masked, device)
    with torch.no_grad():
        logits = model(
            input_ids=input_ids, attention_mask=attention_mask, is_causal=False
        ).logits[0]
    log_probs = F.log_softmax(logits, dim=-1)
    return [log_probs[position, token_ids[position]].item() for position in positions]


def run_one_context(
    model,
    tokenizer,
    genes: list[dict],
    genome_seq: str,
    context_bp: int = 8192,
    device: str = "cuda:0",
) -> list[dict]:
    validate_paper_protocol(context_bp)
    if tokenizer.mask_token_id is None:
        raise ValueError("The tokenizer must provide mask_token_id.")
    results = []
    for gene in tqdm(genes, desc="PhaGen/8192"):
        inputs = build_wt_and_mut_inputs(
            genome_seq, gene["start"], gene["end"], gene["strand"], context_bp
        )
        wt_tokens = tokenizer.encode(inputs["wt_seq"], add_special_tokens=False)
        mut_tokens = tokenizer.encode(inputs["mut_seq"], add_special_tokens=False)
        wt_positions, mut_positions = get_insert_cassette_pairs(inputs, gene["strand"])
        if len(wt_positions) != 15 or len(mut_positions) != 15:
            raise ValueError(f"Incomplete cassette region for {gene['locus_tag']}.")
        wt_lps = get_masked_logprobs(
            model, wt_tokens, wt_positions, device, tokenizer.mask_token_id
        )
        mut_lps = get_masked_logprobs(
            model, mut_tokens, mut_positions, device, tokenizer.mask_token_id
        )
        wt_score = float(sum(wt_lps) / len(wt_lps))
        mut_score = float(sum(mut_lps) / len(mut_lps))
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
                wt_scored_positions=wt_positions,
                mut_scored_positions=mut_positions,
                n_scored_bases=len(wt_positions),
                scoring_mode="dllm_cassette_masked_marginals",
            )
        )
    return results


def main():
    args = parse_args()
    output_path = build_output_path(
        args.results_dir, args.phage, "phagen", step_id=12800
    )
    ensure_output_available(output_path, force=args.force)
    genome, genes = load_inputs(args)
    from phagen import load_model, load_tokenizer

    model = load_model(
        args.model,
        device=args.device,
        torch_dtype=torch.float32,
        revision=args.revision,
        local_files_only=not args.allow_download,
    )
    tokenizer = load_tokenizer(
        args.model, revision=args.revision, local_files_only=not args.allow_download
    )
    results = run_one_context(model, tokenizer, genes, genome, device=args.device)
    payload = result_payload(
        args,
        "phagen",
        results,
        "masked_marginals_cassette",
        step_id=12800,
        pad_attention_mode="attend_pad",
        bos_eos_mode="none",
    )
    save_evo_style_results(payload, output_path, force=args.force)


if __name__ == "__main__":
    main()
