import argparse
import json
import os
from pathlib import Path

from fasta_io import ROOT, protect


PROMPTS = {"evo1": "|r__Duplodnaviria;k__", "evo2": "|r__Duplodnaviria;;;;;;;|"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(PROMPTS), required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Local Evo2 weight file; Evo1 uses its existing offline cache.",
    )
    parser.add_argument(
        "--cache-dir", type=Path, help="Existing Hugging Face cache root for Evo1."
    )
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument(
        "--length",
        type=int,
        default=131072,
        help="New tokens, excluding the taxonomy prompt.",
    )
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, help="Default: Evo1=6, Evo2=4.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-seqlen", type=int, default=140000)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.batch_size = args.batch_size or (6 if args.model == "evo1" else 4)
    if (
        min(args.samples, args.length, args.chunk_size, args.batch_size) < 1
        or args.samples % args.chunk_size
    ):
        parser.error("Positive sizes and a whole number of chunks are required.")
    output_dir = args.output_dir or ROOT / "data/raw_generation" / args.model
    raw, parameters = output_dir / "raw_sequences.jsonl", output_dir / "parameters.json"
    protect([raw, parameters], args.force)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.cache_dir:
        os.environ["HF_HOME"] = str(args.cache_dir.resolve())
    import torch

    if args.model == "evo2":
        if args.checkpoint is None:
            parser.error("--checkpoint is required for Evo2.")
        from evo2 import Evo2

        wrapper = Evo2("evo2_7b", local_path=str(args.checkpoint))

        def generate(prompts):
            return list(
                wrapper.generate(
                    prompt_seqs=prompts,
                    n_tokens=args.length,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    cached_generation=True,
                    verbose=0,
                ).sequences
            )
    else:
        from evo import Evo
        from evo.generation import generate as evo_generate

        wrapper = Evo("evo-1-131k-base")
        model, tokenizer = wrapper.model.to("cuda:0"), wrapper.tokenizer
        for part in (model, getattr(model, "backbone", None)):
            if part is not None and hasattr(part, "config"):
                part.config.max_seqlen = args.max_seqlen

        def generate(prompts):
            sequences, _ = evo_generate(
                prompts,
                model,
                tokenizer,
                n_tokens=args.length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                batched=True,
                cached_generation=True,
                device="cuda:0",
                verbose=0,
            )
            return list(sequences)

    output_dir.mkdir(parents=True, exist_ok=True)
    with raw.open("w") as handle:
        for chunk in range(args.samples // args.chunk_size):
            for batch_index, offset in enumerate(
                range(0, args.chunk_size, args.batch_size)
            ):
                count = min(args.batch_size, args.chunk_size - offset)
                seed = args.seed + chunk * 1000 + batch_index
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                sequences = generate([PROMPTS[args.model]] * count)
                if len(sequences) != count:
                    raise ValueError(
                        "The native generator returned an unexpected sequence count."
                    )
                for row, sequence in enumerate(sequences):
                    handle.write(
                        json.dumps(
                            {
                                "sequence_index": chunk * args.chunk_size
                                + offset
                                + row,
                                "chunk": chunk,
                                "seed": seed,
                                "raw": sequence,
                            }
                        )
                        + "\n"
                    )
                handle.flush()
    parameters.write_text(
        json.dumps(
            {
                "model": "evo-1-131k-base" if args.model == "evo1" else "evo2_7b",
                "prompt": PROMPTS[args.model],
                "new_tokens": args.length,
                "samples": args.samples,
                "chunk_size": args.chunk_size,
                "batch_size": args.batch_size,
                "seed_base": args.seed,
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "stop_at_eos": False,
                "cached_generation": True,
                **({"max_seqlen": args.max_seqlen} if args.model == "evo1" else {}),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
