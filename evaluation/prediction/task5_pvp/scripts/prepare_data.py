#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def read_table(path, delimiter="\t"):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        return list(reader.fieldnames or []), list(reader)


def quality_ok(row):
    try:
        length = int(row["dna_len"])
        aa_length = int(row["aa_len"])
        ambiguous = int(row["has_non_atcg"])
    except (ValueError, TypeError):
        return False
    dna = row["dna_seq"].upper()
    return bool(dna) and length == len(dna) and aa_length > 0 and ambiguous == 0 and not set(dna) - set("ACGT") and length % 3 == 0 and length <= 8192


def remove_aa_conflicts(rows):
    groups = defaultdict(list)
    for row in rows:
        sequence = "".join(row["aa_seq"].split()).upper().rstrip("*")
        if not sequence:
            raise ValueError(f"Empty protein sequence: {row['accession']}")
        groups[sequence].append(row)
    conflicts = {sequence for sequence, group in groups.items() if len({row["label"] for row in group}) > 1}
    return [row for row in rows if "".join(row["aa_seq"].split()).upper().rstrip("*") not in conflicts]


def deduplicate_dna(rows):
    groups = defaultdict(list)
    for row in rows:
        dna = "".join(row["dna_seq"].split()).upper()
        if not dna:
            raise ValueError(f"Empty CDS: {row['accession']}")
        groups[dna].append(row)
    retained = []
    for group in groups.values():
        if len({row["label"] for row in group}) != 1 or len({row["split"] for row in group}) != 1:
            raise ValueError("Identical CDSs have conflicting labels or official partitions")
        retained.append(group[0])
    return retained


def build_benchmark(raw_dir):
    official = {}
    for split in ("test", "train"):
        _, labels = read_table(raw_dir / f"{split}_40.csv", delimiter=",")
        for row in labels:
            accession = row["accession"].strip()
            pair = (split, row["label"].strip())
            if accession in official and official[accession] != pair:
                raise ValueError(f"Conflicting official labels: {accession}")
            official[accession] = pair
    fields, rows = read_table(raw_dir / "threshold40_cds_mapping.tsv")
    if any(row["similarity_threshold"] != "40" for row in rows):
        raise ValueError("The mapping must contain only threshold-40 records")
    for row in rows:
        if official.get(row["accession"]) != (row["split"], row["label"]):
            raise ValueError(f"CDS mapping differs from official labels: {row['accession']}")
    clean = [row for row in rows if quality_ok(row)]
    consistent = remove_aa_conflicts(clean)
    final = deduplicate_dna(consistent)
    if len(final) != 14901 or sum(row["split"] == "train" for row in final) != 7822:
        raise ValueError("The source files do not produce the 14,901-record paper benchmark")
    aa_splits = defaultdict(set)
    for row in final:
        aa_splits[row["aa_seq"]].add(row["split"])
    if any(len(splits) > 1 for splits in aa_splits.values()):
        raise ValueError("Identical protein sequences cross the official split")
    return fields, final, {"mapped": len(rows), "quality_retained": len(clean),
                           "conflicting_protein_rows_removed": len(clean) - len(consistent),
                           "duplicate_cds_rows_removed": len(consistent) - len(final)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/phavip_threshold40.tsv")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {args.output}; use --force")
    fields, rows, counts = build_benchmark(args.raw_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(counts)
    print(args.output)


if __name__ == "__main__":
    main()
