import argparse
import csv
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generation"))
from fasta_io import ROOT, protect


PROMPTS = {"evo1": "|r__Duplodnaviria;k__", "evo2": "|r__Duplodnaviria;;;;;;;|"}
DISPLAY = {"phagen": "PhaGen", "megadna": "megaDNA", "evo1": "Evo1", "evo2": "Evo2"}


def require_single_gpu(torch):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Expose exactly one CUDA GPU with CUDA_VISIBLE_DEVICES for the single-GPU latency protocol."
        )


def native_generator(args, torch):
    if args.model == "phagen":
        from phagen import MDMGenerationConfig, load_model, load_tokenizer

        tokenizer = load_tokenizer(args.checkpoint, local_files_only=True)
        model = load_model(
            args.checkpoint,
            device="cuda:0",
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        inputs = torch.full(
            (1, 1), tokenizer.bos_token_id, dtype=torch.long, device="cuda:0"
        )

        def generate(length, smoke=False):
            config = MDMGenerationConfig(
                mask_token_id=tokenizer.mask_token_id,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=length - 1,
                steps=4 if smoke else 256,
                temperature=1.1,
                top_p=0.9,
                top_k=4,
                alg="entropy",
                alg_temp=0.1,
                num_return_sequences=1,
                pad_attention_mode="attend_pad",
                return_dict_in_generate=True,
                output_history=False,
            )
            return model.diffusion_generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                generation_config=config,
            ).sequences

        return generate, 1, True, "bfloat16"
    if args.model == "megadna":
        if args.model_repo:
            sys.path.insert(0, str(args.model_repo.resolve()))
        model = torch.load(
            args.checkpoint, map_location="cuda:0", weights_only=False
        ).eval()
        import megaDNA.megadna as implementation

        implementation.tqdm = lambda iterable, *a, **k: iterable
        primer = torch.randint(
            1, 5, (1, 4), generator=torch.Generator().manual_seed(0)
        ).to("cuda:0")

        def generate(length, smoke=False):
            return model.generate(
                primer, seq_len=length, temperature=0.95, filter_thres=0.0
            )

        return generate, 4, True, "checkpoint native dtype"
    if args.model == "evo1":
        from evo import Evo
        from evo.generation import Generator

        wrapper = Evo("evo-1-131k-base", device="cuda:0")
        model, tokenizer = wrapper.model.eval(), wrapper.tokenizer
        model.config.max_seqlen = max(140000, args.length + 1)
    else:
        from evo2 import Evo2
        from vortex.model.generation import Generator

        wrapper = Evo2("evo2_7b", local_path=args.checkpoint)
        model, tokenizer = wrapper.model.eval(), wrapper.tokenizer
    inputs = torch.tensor(
        [tokenizer.tokenize(PROMPTS[args.model])], dtype=torch.long, device="cuda:0"
    )
    prompt_tokens = inputs.shape[1]
    generator = Generator(model, tokenizer, temperature=0.7, top_k=4, top_p=1.0)

    def generate(length, smoke=False):
        output = generator.generate(
            input_ids=inputs,
            num_tokens=length - prompt_tokens,
            cached_generation=True,
            device="cuda:0",
            print_generation=False,
            verbose=False,
            stop_at_eos=False,
        )
        return output[0]

    return generate, prompt_tokens, False, "native loader mixed precision"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(DISPLAY), required=True)
    parser.add_argument(
        "--checkpoint", help="Local PhaGen/megaDNA directory or Evo2 weight file."
    )
    parser.add_argument(
        "--model-repo", type=Path, help="Local megaDNA source providing generate."
    )
    parser.add_argument(
        "--cache-dir", type=Path, help="Existing Hugging Face cache root for Evo1."
    )
    parser.add_argument("--length", type=int, default=131072)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/timing")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.model != "evo1" and not args.checkpoint:
        parser.error("--checkpoint is required for this model.")
    if (
        not 2048 <= args.length <= 131072
        or min(args.warmups, args.repeats) < 1
        or args.seed < args.warmups + 2
    ):
        parser.error(
            "Require 2048–131072 total tokens, positive repeats/warmups, and a sufficiently large seed."
        )
    observations = args.output_dir / f"{args.model}_observations.tsv"
    summary_path = args.output_dir / f"{args.model}_summary.tsv"
    parameters = args.output_dir / f"{args.model}_parameters.json"
    protect(
        [
            observations,
            summary_path,
            parameters,
            args.output_dir / "observations.tsv",
            args.output_dir / "summary.tsv",
        ],
        args.force,
    )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.cache_dir:
        os.environ["HF_HOME"] = str(args.cache_dir.resolve())
    import numpy as np
    import torch

    require_single_gpu(torch)
    generate, prompt_tokens, includes_prompt, precision = native_generator(args, torch)
    rows = []
    calls = [("smoke", 0, 2048, args.seed - 2)]
    calls += [
        ("warmup", index, args.length, args.seed - 1 - index)
        for index in range(args.warmups)
    ]
    calls += [
        ("measured", index, args.length, args.seed + index)
        for index in range(args.repeats)
    ]
    for phase, repetition, length, seed in calls:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = generate(length, smoke=phase == "smoke")
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
        expected = length if includes_prompt else length - prompt_tokens
        if tuple(output.shape) != (1, expected):
            raise ValueError(f"Unexpected native token shape: {tuple(output.shape)}")
        rows.append(
            dict(
                model=DISPLAY[args.model],
                phase=phase,
                repetition=repetition + 1,
                seed=seed,
                total_tokens=length,
                prompt_tokens=prompt_tokens,
                new_tokens=length - prompt_tokens,
                output_tokens=expected,
                seconds=seconds,
                new_tokens_per_second=(length - prompt_tokens) / seconds,
            )
        )
    measured = [row["seconds"] for row in rows if row["phase"] == "measured"]
    summary = dict(
        model=DISPLAY[args.model],
        n=len(measured),
        mean_seconds=statistics.mean(measured),
        sd_seconds=statistics.stdev(measured) if len(measured) > 1 else 0.0,
        median_seconds=statistics.median(measured),
        min_seconds=min(measured),
        max_seconds=max(measured),
        total_tokens=args.length,
        new_tokens=args.length - prompt_tokens,
        pooled_new_tokens_per_second=len(measured)
        * (args.length - prompt_tokens)
        / sum(measured),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path, data in [(observations, rows), (summary_path, [summary])]:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(data)
    parameters.write_text(
        json.dumps(
            {
                "model": DISPLAY[args.model],
                **({"method": "generate"} if args.model == "megadna" else {}),
                "gpu_name": torch.cuda.get_device_name(0),
                "precision": precision,
                "batch_size": 1,
                "length_includes_prompt": True,
                "warmups": args.warmups,
                "repeats": args.repeats,
                "seed": args.seed,
                "timer": "perf_counter with CUDA synchronization",
                "scope": "native generation including cache allocation/prefill/sampling, excluding loading/tokenization/decoding/output writing",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
