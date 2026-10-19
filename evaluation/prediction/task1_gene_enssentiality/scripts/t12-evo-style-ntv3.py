#!/usr/bin/env python3


from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

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


_MODELS = ["NTv3_8M_pre", "NTv3_100M_pre", "NTv3_650M_pre"]
_VALID_BASES = {"A", "C", "G", "T", "N"}


def parse_args():
    parser = make_parser('NTv3 masked-region scores for zero-shot phage gene essentiality.')
    parser.add_argument(
        "--model-name",
        choices=_MODELS + [f"InstaDeepAI/{name}" for name in _MODELS],
        default="NTv3_100M_pre",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional local directory for the selected NTv3 checkpoint.",
    )
    parser.add_argument("--pad-to-multiple-of", type=int, choices=[128], default=128)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def load_ntv3(repo_id: str, device: torch.device, local_files_only: bool):
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        repo_id, trust_remote_code=True, local_files_only=local_files_only
    )
    model = AutoModelForMaskedLM.from_pretrained(
        repo_id,
        trust_remote_code=True,
        local_files_only=local_files_only,
        torch_dtype=torch.float32,
    )
    return tokenizer, model.to(device).eval()


def validate_ntv3_sequence(seq: str) -> str:
    seq = seq.upper()
    bad = sorted(set(seq) - _VALID_BASES)
    if bad:
        raise ValueError(f"NTv3 only supports A/C/G/T/N; found {bad}")
    return seq


def tokenize_sequence(
    seq: str, tokenizer, device: torch.device, pad_to_multiple_of: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    seq = validate_ntv3_sequence(seq)
    batch = tokenizer(
        [seq],
        add_special_tokens=False,
        padding=True,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors="pt",
    )
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(device)
    n_pad = int(input_ids.shape[1] - len(seq))
    return (input_ids, attention_mask, n_pad)


@torch.inference_mode()
def masked_logprob_mean(
    model,
    tokenizer,
    seq: str,
    positions: list[int],
    device: torch.device,
    pad_to_multiple_of: int,
) -> tuple[float, int, int]:
    if not positions:
        return (float("nan"), 0, 0)
    (input_ids, attention_mask, n_pad) = tokenize_sequence(
        seq, tokenizer, device, pad_to_multiple_of
    )
    pos_tensor = torch.as_tensor(positions, dtype=torch.long, device=device)
    true_ids = input_ids[0, pos_tensor]
    masked_ids = input_ids.clone()
    masked_ids[0, pos_tensor] = tokenizer.mask_token_id
    param_dtype = next(model.parameters()).dtype
    use_autocast = device.type == "cuda" and param_dtype in {
        torch.float16,
        torch.bfloat16,
    }
    amp_context = (
        torch.autocast(device_type="cuda", dtype=param_dtype)
        if use_autocast
        else nullcontext()
    )
    with amp_context:
        out = model(input_ids=masked_ids, attention_mask=attention_mask)
    logits = out["logits"] if isinstance(out, dict) else out.logits
    log_probs = F.log_softmax(logits[0, pos_tensor, :].float(), dim=-1)
    selected = log_probs.gather(1, true_ids[:, None]).squeeze(1)
    return (float(selected.mean().item()), len(positions), n_pad)


def run_one_context(
    model,
    tokenizer,
    genes: list[dict],
    genome_seq: str,
    context_bp: int = 8192,
    device: torch.device = torch.device("cpu"),
    pad_to_multiple_of: int = 128,
) -> list[dict]:
    validate_paper_protocol(context_bp)
    results = []
    for gene in tqdm(genes, desc="NTv3/8192"):
        inputs = build_wt_and_mut_inputs(
            genome_seq, gene["start"], gene["end"], gene["strand"], context_bp
        )
        wt_positions, mut_positions = get_insert_cassette_pairs(inputs, gene["strand"])
        if len(wt_positions) != 15 or len(mut_positions) != 15:
            raise ValueError(f"Incomplete cassette region for {gene['locus_tag']}.")
        wt_score, n_wt, wt_pad = masked_logprob_mean(
            model, tokenizer, inputs["wt_seq"], wt_positions, device, pad_to_multiple_of
        )
        mut_score, n_mut, mut_pad = masked_logprob_mean(
            model,
            tokenizer,
            inputs["mut_seq"],
            mut_positions,
            device,
            pad_to_multiple_of,
        )
        results.append(
            gene_result(
                gene,
                inputs,
                wt_score,
                mut_score,
                wt_scored_positions=wt_positions,
                mut_scored_positions=mut_positions,
                n_scored_bases=n_wt,
                n_scored_bases_mut=n_mut,
                wt_n_pad=wt_pad,
                mut_n_pad=mut_pad,
                scoring_mode="ntv3_cassette_masked_marginals",
            )
        )
    return results


def main():
    args = parse_args()
    name = args.model_name.rsplit("/", 1)[-1]
    output_path = build_output_path(args.results_dir, args.phage, name, step_id="hf")
    ensure_output_available(output_path, force=args.force)
    genome, genes = load_inputs(args)
    genome = validate_ntv3_sequence(genome)
    repo_id = (
        str(args.model_path.expanduser().resolve())
        if args.model_path is not None
        else f"InstaDeepAI/{name}"
    )
    device = torch.device(args.device)
    tokenizer, model = load_ntv3(
        repo_id, device, local_files_only=not args.allow_download
    )
    if tokenizer.mask_token_id is None:
        raise ValueError("The tokenizer must provide mask_token_id.")
    results = run_one_context(
        model,
        tokenizer,
        genes,
        genome,
        device=device,
        pad_to_multiple_of=args.pad_to_multiple_of,
    )
    payload = result_payload(
        args,
        name,
        results,
        "ntv3_masked_marginals_cassette",
        step_id="hf",
        bos_eos_mode="none_ntv3",
        ntv3_pad_to_multiple_of=128,
        ntv3_pad_mode="hf_pad_token_with_attention_mask",
        torch_dtype="torch.float32",
    )
    save_evo_style_results(
        payload,
        output_path,
        force=args.force,
    )


if __name__ == "__main__":
    main()
