import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from fasta_io import ROOT, header_fields, protect, read_fasta, write_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-fasta", type=Path, default=ROOT / "data/cohorts/natural.fasta"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/cohorts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-records", type=int, default=10000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    paths = [
        args.output_dir / name
        for name in [
            "scrambled.fasta",
            "scrambled_mapping.tsv",
            "scrambled_sampling.json",
        ]
    ]
    protect(paths, args.force)
    if args.source_fasta.resolve() == paths[0].resolve():
        parser.error("Source and output FASTA must differ.")
    if args.seed < 0 or args.expected_records < 1:
        parser.error("Nonnegative seed and positive expected count are required.")
    fields = [
        "scrambled_id",
        "natural_id",
        "record_index",
        "source",
        "length_bp",
        "gc_fraction",
        "a_count",
        "c_count",
        "g_count",
        "t_count",
        "natural_sequence_sha256",
        "scrambled_sequence_sha256",
    ]
    rows, sequences = [], []
    for index, (header, dna) in enumerate(read_fasta(args.source_fasta)):
        natural_id = header.split()[0]
        counts = [dna.count(base) for base in "ACGT"]
        if not dna or sum(counts) != len(dna):
            raise ValueError(f"Empty or non-ACGT sequence: {natural_id}")
        array = np.frombuffer(dna.encode("ascii"), dtype=np.uint8).copy()
        rng = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence([args.seed, index]))
        )
        rng.shuffle(array)
        shuffled = array.tobytes().decode("ascii")
        if [shuffled.count(base) for base in "ACGT"] != counts:
            raise RuntimeError("Base composition changed during permutation.")
        identifier = natural_id + "_scrambled"
        sequences.append((identifier, natural_id, shuffled))
        gc = (counts[1] + counts[2]) / len(dna)
        rows.append(
            dict(
                zip(
                    fields,
                    [
                        identifier,
                        natural_id,
                        index,
                        header_fields(header).get("source", "unspecified"),
                        len(dna),
                        format(gc, ".12g"),
                        *counts,
                        hashlib.sha256(dna.encode("ascii")).hexdigest(),
                        hashlib.sha256(shuffled.encode("ascii")).hexdigest(),
                    ],
                )
            )
        )
    if len(rows) != args.expected_records or len(
        {row["natural_id"] for row in rows}
    ) != len(rows):
        raise ValueError(
            "Natural record count or unique identifiers differ from the expected cohort."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with paths[0].open("w") as handle:
        for identifier, natural_id, sequence in sequences:
            write_record(
                handle,
                identifier,
                sequence,
                {
                    "natural_id": natural_id,
                    "control": "mononucleotide_shuffle",
                    "length": len(sequence),
                },
            )
    with paths[1].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    paths[2].write_text(
        json.dumps(
            {
                "seed": args.seed,
                "algorithm": "mononucleotide permutation without replacement",
                "rng": "numpy.random.Generator(PCG64(SeedSequence([seed, zero_based_record_index])))",
                "numpy_version": np.__version__,
                "n": len(rows),
                "total_bp": sum(row["length_bp"] for row in rows),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
