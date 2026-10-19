#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path


TASK_DIR = Path(__file__).resolve().parent.parent
FIELDS = [
    "embedding_row",
    "record_id",
    "source_index",
    "genome_id",
    "seq_id",
    "annotation",
    "host",
    "dna_len",
    "dna",
    "protein",
]
UNKNOWN_VALUES = {"", "unknown", "na", "n/a", "none", "null"}


def prepare_rows(input_path: Path) -> tuple[list[dict], list[dict]]:
    with input_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 12509:
        raise ValueError("Expected 12,509 source DRTail records.")
    benchmark, embedding = [], []
    for source_index, row in enumerate(rows):
        dna = "".join(row["dna"].split()).upper()
        if not dna or set(dna) - set("ACGT"):
            raise ValueError(f"Invalid CDS sequence at source row {source_index}.")
        if len(dna) % 3:
            continue
        embedding_row = len(embedding)
        embedding.append({"record_id": embedding_row, "dna_seq": dna})
        if row["genome_id"].strip().lower() in UNKNOWN_VALUES:
            continue
        benchmark.append(
            {
                "embedding_row": embedding_row,
                "record_id": len(benchmark),
                "source_index": source_index,
                "genome_id": row["genome_id"],
                "seq_id": row["seq_id"],
                "annotation": row["annotation"],
                "host": row["host"],
                "dna_len": len(dna),
                "dna": dna,
                "protein": row["protein"],
            }
        )
    if len(embedding) != 12506 or len(benchmark) != 12236:
        raise ValueError("Unexpected triplet-length or genome-resolved cohort size.")
    if len({row["genome_id"] for row in benchmark}) != 2214:
        raise ValueError("Expected 2,214 source genomes.")
    for field in ["dna", "protein"]:
        if len({row[field] for row in benchmark}) != len(benchmark):
            raise ValueError(f"Duplicate {field} sequences in the benchmark.")
    return benchmark, embedding


def write_table(
    path: Path, rows: list[dict], fields: list[str], delimiter: str, force: bool
) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=fields, delimiter=delimiter, lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    contents = buffer.getvalue()
    if path.exists() and not force:
        if path.read_text() == contents:
            return
        raise FileExistsError(
            f"Output already exists: {path}; use --force to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=TASK_DIR / "data/DRTail.csv")
    parser.add_argument("--output", type=Path, default=TASK_DIR / "data/benchmark.csv")
    parser.add_argument(
        "--embedding-output",
        type=Path,
        help="Optional 12,506-row TSV for frozen embedding extraction.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    benchmark, embedding = prepare_rows(args.input)
    write_table(args.output, benchmark, FIELDS, ",", args.force)
    if args.embedding_output:
        write_table(
            args.embedding_output, embedding, ["record_id", "dna_seq"], "\t", args.force
        )
    print(
        f"Prepared {len(benchmark):,} CDSs from 2,214 genomes; {len(embedding):,} embedding inputs."
    )


if __name__ == "__main__":
    main()
