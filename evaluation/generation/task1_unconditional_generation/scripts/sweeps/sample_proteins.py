import argparse
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generation"))
from fasta_io import protect, read_fasta


def select_proteins(spans_path, proteins_path, tag):
    spans = [
        sequence for _, sequence in read_fasta(spans_path) if len(sequence) >= 2000
    ]
    proteins = [sequence.rstrip("*") for _, sequence in read_fasta(proteins_path)]
    proteins = [sequence for sequence in proteins if 0 < len(sequence) < 1026]
    if not spans or not proteins:
        return []
    rng = random.Random(42)
    rng.sample(spans, min(50, len(spans)))
    selected = rng.sample(proteins, min(100, len(proteins)))
    return [(f"{tag}__o{index}", sequence) for index, sequence in enumerate(selected)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spans",
        type=Path,
        required=True,
        help="Configuration extracted-spans FASTA in its original order.",
    )
    parser.add_argument(
        "--proteins",
        type=Path,
        required=True,
        help="Full geNomad annotate proteins FASTA for that configuration.",
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    protect([args.output], args.force)
    if args.output.resolve() in {args.spans.resolve(), args.proteins.resolve()}:
        parser.error("Output must differ from the input files.")
    if not args.tag or any(character.isspace() for character in args.tag):
        parser.error("A nonempty tag without whitespace is required.")
    records = select_proteins(args.spans, args.proteins, args.tag)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for identifier, sequence in records:
            handle.write(f">{identifier}\n{sequence}\n")
    print(f"Saved {len(records)} sampled proteins to {args.output}")


if __name__ == "__main__":
    main()
