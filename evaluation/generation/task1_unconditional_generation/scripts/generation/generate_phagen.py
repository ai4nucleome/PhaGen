import argparse
import json
from pathlib import Path

from fasta_io import ROOT, protect, seed_torch, write_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local PhaGen checkpoint.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float32", "float16"], default="bfloat16"
    )
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--length", type=int, default=131072, help="Total canvas including BOS."
    )
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument(
        "--alg",
        choices=["entropy", "topk_margin", "maskgit_plus", "origin", "p2"],
        default="entropy",
    )
    parser.add_argument("--alg-temp", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/raw_generation/phagen"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if min(args.samples, args.batch_size, args.steps) < 1 or args.length < 2:
        parser.error(
            "Samples, batch size and steps must be positive; length must exceed one."
        )
    raw = args.output_dir / "raw_canvases.fasta"
    config_path = args.output_dir / "parameters.json"
    protect([raw, config_path], args.force)

    import torch
    from phagen import MDMGenerationConfig, load_model, load_tokenizer

    seed_torch(torch, args.seed)
    tokenizer = load_tokenizer(args.model, local_files_only=True)
    model = load_model(
        args.model,
        device=args.device,
        torch_dtype=getattr(torch, args.dtype),
        local_files_only=True,
    )
    config = MDMGenerationConfig(
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=args.length - 1,
        steps=args.steps,
        temperature=args.temperature,
        top_p=None if args.top_p >= 1 else args.top_p,
        top_k=None if args.top_k <= 0 else args.top_k,
        alg=args.alg,
        alg_temp=args.alg_temp,
        num_return_sequences=1,
        block_pad_token=False,
        pad_attention_mode="attend_pad",
        return_dict_in_generate=True,
        output_history=False,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with raw.open("w") as handle:
        for start in range(0, args.samples, args.batch_size):
            count = min(args.batch_size, args.samples - start)
            inputs = torch.full(
                (count, 1), tokenizer.bos_token_id, dtype=torch.long, device=args.device
            )
            with torch.no_grad():
                output = model.diffusion_generate(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    generation_config=config,
                )
            for offset, tokens in enumerate(output.sequences.tolist()):
                index = start + offset
                text = tokenizer.decode(
                    tokens,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                write_record(
                    handle,
                    f"seq{index}",
                    text,
                    {
                        "sequence_index": index,
                        "prompt_index": index,
                        "return_index": 0,
                        "prompt": "",
                        "prompt_with_bos": tokenizer.bos_token,
                        "canvas_length": len(tokens),
                    },
                )
    parameters = {
        key: value
        for key, value in vars(args).items()
        if key not in {"model", "output_dir", "device", "force"}
    }
    parameters.update(
        model="PhaGen",
        prompt="BOS",
        pad_attention_mode="attend_pad",
        block_pad_token=False,
    )
    config_path.write_text(json.dumps(parameters, indent=2) + "\n")


if __name__ == "__main__":
    main()
