#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
from common import (
    TASK_DIR,
    COHORTS,
    GENERATED,
    read_fasta,
    write_table,
    ensure_available,
)

SEQUENCE_FIELDS = [
    "group",
    "seq_name",
    "length_bp",
    "gc_bases",
    "gc_pct",
    "orf_count",
    "orf_density_per_kb",
    "summed_orf_length_bp",
    "coding_density",
    "coding_density_pct",
]
ORF_FIELDS = ["group", "seq_name", "orf_id", "orf_length_bp"]


def collect_cohort(key, fasta, genes_path):
    sequences = {}
    for name, dna in read_fasta(fasta):
        if name in sequences or len(dna) < 2000 or set(dna) - set("ACGT"):
            raise ValueError(f"Invalid retained input: {name}")
        sequences[name] = {
            "length": len(dna),
            "gc": dna.count("G") + dna.count("C"),
            "orfs": 0,
            "coding": 0,
        }
    if not sequences:
        raise ValueError(f"Empty FASTA: {fasta}")
    seen = set()
    orfs = []
    with genes_path.open(newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            gene = row["gene"]
            parent = gene.rsplit("_", 1)[0]
            if gene in seen or parent not in sequences:
                raise ValueError(f"Duplicate or unmapped full-input gene: {gene}")
            seen.add(gene)
            start, end = int(row["start"]), int(row["end"])
            if not 1 <= start <= end <= sequences[parent]["length"]:
                raise ValueError(f"Invalid gene coordinates: {gene}")
            length = end - start + 1
            if length != int(row["length"]):
                raise ValueError(
                    f"Gene length differs from its inclusive coordinates: {gene}"
                )
            sequences[parent]["orfs"] += 1
            sequences[parent]["coding"] += length
            orfs.append(
                {
                    "group": COHORTS.get(key, key),
                    "seq_name": parent,
                    "orf_id": gene,
                    "orf_length_bp": length,
                }
            )
    rows = []
    for name, seq in sequences.items():
        length, gc, count, coding = seq["length"], seq["gc"], seq["orfs"], seq["coding"]
        rows.append(
            {
                "group": COHORTS.get(key, key),
                "seq_name": name,
                "length_bp": length,
                "gc_bases": gc,
                "gc_pct": 100 * gc / length,
                "orf_count": count,
                "orf_density_per_kb": count * 1000 / length,
                "summed_orf_length_bp": coding,
                "coding_density": coding / length,
                "coding_density_pct": 100 * coding / length,
            }
        )
    return rows, orfs


def describe(group, metric, values, unit):
    values = np.asarray(values)
    return {
        "group": group,
        "metric": metric,
        "n": len(values),
        "unit": unit,
        "minimum": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "mean": float(np.mean(values)),
        "maximum": float(np.max(values)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, default=TASK_DIR / "data/cohorts")
    parser.add_argument(
        "--annotation-dir", type=Path, default=TASK_DIR / "data/annotations"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=TASK_DIR / "results/sequence_statistics"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    names = ["per_sequence.tsv", "per_orf.tsv", "summary.tsv", "retained_counts.tsv"]
    ensure_available([args.output_dir / name for name in names], args.force)
    all_rows, all_orfs, summaries, counts = [], [], [], []
    for key in COHORTS:
        rows, orfs = collect_cohort(
            key,
            args.sequence_dir / f"{key}.fasta",
            args.annotation_dir / key / "genes.tsv",
        )
        all_rows.extend(rows)
        all_orfs.extend(orfs)
        for metric in [
            "length_bp",
            "gc_pct",
            "orf_count",
            "orf_density_per_kb",
            "coding_density",
        ]:
            summaries.append(
                describe(
                    COHORTS.get(key, key),
                    metric,
                    [row[metric] for row in rows],
                    "input sequence",
                )
            )
        summaries.append(
            describe(
                COHORTS.get(key, key),
                "orf_length_bp",
                [row["orf_length_bp"] for row in orfs],
                "predicted ORF",
            )
        )
        if key in GENERATED:
            counts.append(
                {"group": COHORTS.get(key, key), "retained_sequences": len(rows)}
            )
    write_table(args.output_dir / names[0], all_rows, SEQUENCE_FIELDS)
    write_table(args.output_dir / names[1], all_orfs, ORF_FIELDS)
    write_table(args.output_dir / names[2], summaries)
    write_table(args.output_dir / names[3], counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
