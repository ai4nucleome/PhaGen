import argparse
import csv
import math
from pathlib import Path
import re
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generation"))
from fasta_io import protect, read_fasta


def summarize(fasta, genes, viruses, features=None):
    inputs = {}
    for header, sequence in read_fasta(fasta):
        name = header.split()[0]
        if name in inputs:
            raise ValueError(f"Duplicate input sequence: {name}")
        inputs[name] = sequence
    sequences = {
        name: sequence
        for name, sequence in inputs.items()
        if len(sequence) >= 2000 and not set(sequence) - set("ACGT")
    }
    if not sequences:
        return dict(
            n_spans=0,
            viral=0,
            viral_rate=math.nan,
            ci_lo=math.nan,
            ci_hi=math.nan,
            gc_med=math.nan,
            coding_med=math.nan,
            span_med_bp=math.nan,
        )
    if genes is None or viruses is None:
        raise ValueError("Nonempty retained inputs require both --genes and --viruses.")
    coding = {name: 0 for name in sequences}
    seen = set()
    with genes.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            identifier = row["gene"]
            parent = identifier.rsplit("_", 1)[0]
            if parent not in sequences:
                continue
            if identifier in seen:
                raise ValueError(f"Duplicate gene: {identifier}")
            seen.add(identifier)
            start, end = int(row["start"]), int(row["end"])
            if not 1 <= start <= end <= len(
                sequences[parent]
            ) or end - start + 1 != int(row["length"]):
                raise ValueError(f"Invalid inclusive ORF coordinates: {identifier}")
            coding[parent] += end - start + 1
    coding_values = {
        name: float(f"{coding[name] / len(sequence):.4f}")
        for name, sequence in sequences.items()
    }
    if features is not None:
        reported = {}
        with features.open() as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                name = row["seq_name"]
                if name not in sequences:
                    continue
                if name in reported:
                    raise ValueError(f"Duplicate feature record: {name}")
                value = float(row["coding_density"])
                if (
                    not math.isfinite(value)
                    or abs(value - coding_values[name]) > 0.000050001
                ):
                    raise ValueError(
                        f"Coding-density feature disagrees with gene coordinates: {name}"
                    )
                reported[name] = value
        if reported.keys() != sequences.keys():
            raise ValueError("Features must cover every retained input exactly once.")
        coding_values = reported
    viral = set()
    regions = set()
    with viruses.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            name = row["seq_name"]
            if name in regions:
                raise ValueError(f"Duplicate viral region: {name}")
            regions.add(name)
            if name in inputs:
                parent, length = name, len(inputs[name])
            else:
                match = re.fullmatch(r"(.+)\|provirus_(\d+)_(\d+)", name)
                if match is None:
                    raise ValueError(f"Unmapped viral region: {name}")
                parent, start, end = match[1], int(match[2]), int(match[3])
                if parent not in inputs or not 1 <= start <= end <= len(inputs[parent]):
                    raise ValueError(f"Invalid provirus coordinates: {name}")
                length = end - start + 1
            if int(row["length"]) != length:
                raise ValueError(f"Invalid viral region length: {name}")
            if parent in sequences:
                viral.add(parent)
    n, k = len(sequences), len(viral)
    rate = k / n if n else math.nan
    if n:
        z = 1.959963984540054
        denominator = 1 + z * z / n
        center = (rate + z * z / (2 * n)) / denominator
        radius = (
            z / denominator * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n))
        )
        low, high = center - radius, center + radius
    else:
        low = high = math.nan
    return dict(
        n_spans=n,
        viral=k,
        viral_rate=rate,
        ci_lo=low,
        ci_hi=high,
        gc_med=statistics.median(
            (sequence.count("G") + sequence.count("C")) / len(sequence)
            for sequence in sequences.values()
        )
        if n
        else math.nan,
        coding_med=statistics.median(coding_values.values()) if n else math.nan,
        span_med_bp=statistics.median(map(len, sequences.values())) if n else math.nan,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument(
        "--genes",
        type=Path,
        help="geNomad full-input annotate genes TSV; required for nonempty inputs.",
    )
    parser.add_argument(
        "--viruses",
        type=Path,
        help="geNomad virus_summary TSV; required for nonempty inputs.",
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--features",
        type=Path,
        help="geNomad marker-classification features TSV; use its reported precision for coding-density medians.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    protect([args.output], args.force)
    if args.output.resolve() in {
        path.resolve()
        for path in [args.fasta, args.genes, args.viruses]
        if path is not None
    } | ({args.features.resolve()} if args.features is not None else set()):
        parser.error("Output must differ from the inputs.")
    row = {
        "tag": args.tag,
        **summarize(args.fasta, args.genes, args.viruses, args.features),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), delimiter="\t")
        writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
