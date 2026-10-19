import argparse
import json
import sys
from pathlib import Path

from fasta_io import ROOT, protect, seed_torch, write_record


NUCLEOTIDES = ["**", "A", "T", "C", "G", "#"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local megaDNA_phage_145M.pt.")
    parser.add_argument(
        "--model-repo", type=Path, help="Local megaDNA source providing generate."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Default: one batch per seed group, as in the paper.",
    )
    parser.add_argument("--length", type=int, default=131072)
    parser.add_argument("--prime-length", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--filter-thres", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--single-seed",
        action="store_true",
        help="Use one seed for a parameter-sweep cell.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/raw_generation/megadna"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if min(args.samples, args.prime_length) < 1 or args.length <= args.prime_length:
        parser.error(
            "Samples and primer must be positive; length must exceed primer length."
        )
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("Batch size must be positive.")
    raw = args.output_dir / "raw_sequences.fasta"
    parameters = args.output_dir / "parameters.json"
    protect([raw, parameters], args.force)
    if args.single_seed:
        groups = [(args.samples, args.seed)]
    else:
        third = args.samples // 3
        groups = [
            (third, args.seed),
            (third, args.seed + 100),
            (args.samples - 2 * third, args.seed + 200),
        ]
    if args.batch_size and any(count % args.batch_size for count, _ in groups):
        parser.error("Each seed-group size must be divisible by batch size.")
    if args.model_repo:
        sys.path.insert(0, str(args.model_repo.resolve()))
    import torch

    seed_torch(torch, args.seed)
    model = torch.load(args.model, map_location=args.device, weights_only=False).eval()
    if not callable(getattr(model, "generate", None)):
        raise AttributeError(
            "The local megaDNA implementation must provide its native generate method."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = 0
    with raw.open("w") as handle:
        for count, seed in groups:
            if not count:
                continue
            seed_torch(torch, seed)
            batch = args.batch_size or count
            for _ in range(0, count, batch):
                primer = torch.randint(
                    1, 5, (batch, args.prime_length), device=args.device
                )
                with torch.no_grad():
                    tokens = (
                        model.generate(
                            primer,
                            seq_len=args.length,
                            temperature=args.temperature,
                            filter_thres=args.filter_thres,
                        )
                        .long()
                        .cpu()
                    )
                for row in tokens.tolist():
                    decoded = "".join(
                        NUCLEOTIDES[token] if 0 <= token < 6 else "?" for token in row
                    )
                    write_record(
                        handle,
                        f"seq{index}",
                        decoded,
                        {
                            "sequence_index": index,
                            "prime_length": args.prime_length,
                            "canvas_length": len(row),
                        },
                    )
                    index += 1
    parameters.write_text(
        json.dumps(
            {
                "model": "megaDNA_phage_145M",
                "method": "generate",
                "samples": args.samples,
                "seed_groups": [
                    {"samples": n, "seed": s, "batch_size": args.batch_size or n}
                    for n, s in groups
                    if n
                ],
                "context_length": args.length,
                "prime_length": args.prime_length,
                "temperature": args.temperature,
                "filter_thres": args.filter_thres,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
