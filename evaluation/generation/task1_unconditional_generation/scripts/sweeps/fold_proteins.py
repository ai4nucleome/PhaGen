import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generation"))
from fasta_io import protect, read_fasta


def shard_proteins(path, shard_index, num_shards):
    eligible = [
        (header.split()[0], sequence.rstrip("*"))
        for header, sequence in read_fasta(path)
    ]
    eligible = [
        (identifier, sequence)
        for identifier, sequence in eligible
        if 0 < len(sequence) < 1026
    ]
    identifiers = [identifier for identifier, _ in eligible]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Protein IDs must be unique.")
    return [
        record
        for index, record in enumerate(eligible)
        if index % num_shards == shard_index
    ]


def mean_plddt(output):
    scores, mask = output["plddt"][0], output["atom37_atom_exists"][0]
    return ((scores * mask).sum(-1) / mask.sum(-1).clamp(min=1)).float().mean().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proteins", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="facebook/esmfold_v1",
        help="Local checkpoint directory or model ID present in the local cache.",
    )
    parser.add_argument(
        "--cache-dir", type=Path, help="Existing Hugging Face model cache directory."
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Require num-shards > 0 and 0 <= shard-index < num-shards.")
    protect([args.output], args.force)
    if args.output.resolve() == args.proteins.resolve():
        parser.error("Output must differ from the input file.")
    records = shard_proteins(args.proteins, args.shard_index, args.num_shards)

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU with CUDA_VISIBLE_DEVICES.")
    from transformers import AutoTokenizer, EsmForProteinFolding

    options = {"local_files_only": True, "cache_dir": args.cache_dir}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **options)
    model = EsmForProteinFolding.from_pretrained(args.model, **options).cuda().eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["id", "len", "plddt"], delimiter="\t"
        )
        writer.writeheader()
        for identifier, sequence in records:
            try:
                ids = tokenizer(
                    [sequence], return_tensors="pt", add_special_tokens=False
                )["input_ids"].cuda()
                model.trunk.set_chunk_size(64 if len(sequence) > 700 else 128)
                with torch.no_grad():
                    output = model(ids)
                score = f"{mean_plddt(output):.4f}"
            except RuntimeError:
                torch.cuda.empty_cache()
                score = "NA"
                failures += 1
            writer.writerow({"id": identifier, "len": len(sequence), "plddt": score})
            handle.flush()
    if failures:
        raise RuntimeError(
            f"{failures} protein folds failed; their pLDDT is recorded as NA."
        )


if __name__ == "__main__":
    main()
