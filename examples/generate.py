"""Generate token canvases and extract complete DNA contigs."""

import argparse
import math
from pathlib import Path

import torch

from phagen import DEFAULT_MODEL_ID, MDMGenerationConfig, load_model, load_tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--length", type=int, default=256, help="Total token canvas, including the BOS prefix.")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--min-length", type=int, default=1, help="Minimum extracted DNA length; paper cohort: 2000.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/generate"))
    args = parser.parse_args()
    if args.length < 2 or min(args.steps, args.samples, args.min_length) < 1:
        parser.error("length must be >= 2; steps, samples and min-length must be positive.")
    tokens_path = args.output_dir / "tokens.pt"
    fasta_path = args.output_dir / "contigs.fasta"
    if tokens_path.exists() or fasta_path.exists():
        parser.error("Output files already exist; choose a new --output-dir.")

    tokenizer = load_tokenizer(args.model, revision=args.revision, local_files_only=args.local_files_only)
    model = load_model(
        args.model, device=args.device, torch_dtype=getattr(torch, args.dtype),
        revision=args.revision, local_files_only=args.local_files_only,
    )
    if args.length > math.prod(model.max_seq_len):
        parser.error("length exceeds the model's maximum token canvas.")
    torch.manual_seed(args.seed)
    prefix = torch.tensor([[tokenizer.bos_token_id]], device=model.device)
    config = MDMGenerationConfig(
        max_length=args.length, steps=args.steps,
        alg="entropy", alg_temp=0.1, temperature=1.1, top_p=0.90, top_k=4,
        num_return_sequences=args.samples,
        mask_token_id=tokenizer.mask_token_id, pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
    )
    with torch.inference_mode():
        result = model.diffusion_generate(
            input_ids=prefix, attention_mask=torch.ones_like(prefix), generation_config=config,
        )
    tokens = result.sequences.cpu()
    records = []
    for row, canvas in enumerate(tokens, start=1):
        for index, sequence in enumerate(model.extract_contigs(canvas, tokenizer, args.min_length), start=1):
            records.append(f">canvas_{row}_contig_{index}\n{sequence}\n")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(tokens, tokens_path)
    fasta_path.write_text("".join(records))
    print(f"Saved {tokens.shape[0]} token canvas(es) to {tokens_path}")
    print(f"Saved {len(records)} complete contig(s) to {fasta_path}")
    if not records:
        print("No complete contigs passed extraction; this can happen with short canvases.")


if __name__ == "__main__":
    main()

