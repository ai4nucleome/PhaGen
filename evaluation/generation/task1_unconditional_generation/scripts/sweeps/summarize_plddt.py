import argparse
from collections import defaultdict
import csv
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generation"))
from fasta_io import ROOT, protect, read_fasta


def aggregate(scores, sequences):
    lengths = {
        header.split()[0]: len(sequence) for header, sequence in read_fasta(sequences)
    }
    grouped = defaultdict(list)
    seen = set()
    for row in scores:
        identifier = row["id"]
        if (
            identifier in seen
            or identifier not in lengths
            or lengths[identifier] != int(row["len"])
        ):
            raise ValueError(
                f"Duplicate protein or inconsistent input sequence: {identifier}"
            )
        seen.add(identifier)
        value = float(row["plddt"])
        if not 0 <= value <= 1:
            raise ValueError(f"Expected 0–1 pLDDT for {identifier}")
        grouped[identifier.rsplit("__o", 1)[0]].append(value)
    if seen != lengths.keys():
        raise ValueError("The fixed protein sample and score identifiers differ.")
    return [
        {
            "tag": tag,
            "n_folded": len(values),
            "plddt_med": float(f"{statistics.median(values):.4f}") * 100,
        }
        for tag, values in sorted(grouped.items())
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores", type=Path, default=ROOT / "results/sweeps/phagen_protein_plddt.tsv"
    )
    parser.add_argument(
        "--proteins",
        type=Path,
        default=ROOT / "data/sweeps/phagen_parameters/folded_proteins.faa",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    protect([args.output], args.force)
    if args.output.resolve() in {args.scores.resolve(), args.proteins.resolve()}:
        parser.error("Output must differ from the inputs.")
    with args.scores.open() as handle:
        rows = aggregate(list(csv.DictReader(handle, delimiter="\t")), args.proteins)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["tag", "n_folded", "plddt_med"], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
