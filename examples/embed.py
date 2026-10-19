"""Extract paper-style sequence embeddings from a small DNA FASTA."""

import argparse
from pathlib import Path

import torch

from phagen import DEFAULT_MODEL_ID, load_model, load_tokenizer


def read_fasta(path):
    identifiers, sequences = [], []
    seen = set()
    parts = []
    current_id = None

    def finish_record():
        sequence = "".join(parts)
        if not sequence or set(sequence) - set("ACGT"):
            raise ValueError(f"Record {current_id!r} must contain nonempty A/C/G/T DNA.")
        identifiers.append(current_id)
        sequences.append(sequence)

    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    finish_record()
                header = line[1:].split()
                if not header or header[0] in seen:
                    raise ValueError("FASTA headers must have nonempty, unique record IDs.")
                current_id = header[0]
                seen.add(current_id)
                parts = []
            else:
                if current_id is None:
                    raise ValueError("DNA must follow a FASTA header starting with '>'.")
                parts.append(line.upper())
    if current_id is None:
        raise ValueError("The input FASTA contains no records.")
    finish_record()
    return identifiers, sequences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive.")
    if args.output.exists():
        parser.error("Output already exists; choose a new --output path.")
    identifiers, sequences = read_fasta(args.input)
    tokenizer = load_tokenizer(args.model, revision=args.revision, local_files_only=args.local_files_only)
    model = load_model(
        args.model, device=args.device, torch_dtype=getattr(torch, args.dtype),
        revision=args.revision, local_files_only=args.local_files_only,
    )
    vectors = []
    for start in range(0, len(sequences), args.batch_size):
        encoded = tokenizer(
            sequences[start:start + args.batch_size], padding=True,
            add_special_tokens=False, truncation=False, return_tensors="pt",
        ).to(model.device)
        vectors.append(model.embed(encoded["input_ids"], encoded["attention_mask"]).cpu())
    embeddings = torch.cat(vectors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ids": identifiers, "embeddings": embeddings}, args.output)
    print(f"Saved embeddings {tuple(embeddings.shape)} ({embeddings.dtype}) to {args.output}")


if __name__ == "__main__":
    main()

