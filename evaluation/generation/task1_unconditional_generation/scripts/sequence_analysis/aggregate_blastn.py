#!/usr/bin/env python3
"""Merge all reported BLASTN query intervals, retaining no-hit queries at zero."""

from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
from common import (
    TASK_DIR,
    COHORTS,
    GENERATED,
    dna_lengths,
    write_table,
    ensure_available,
)

HSP_FIELDS = [
    "group",
    "seq_name",
    "subject_id",
    "pident",
    "alignment_length",
    "qstart",
    "qend",
    "qlen",
    "query_span_bp",
    "hsp_query_coverage_pct",
    "evalue",
    "bitscore",
]

QUERY_FIELDS = [
    "group",
    "seq_name",
    "length_bp",
    "status",
    "n_targets",
    "n_hsps",
    "covered_bp",
    "merged_query_coverage_pct",
]

SUMMARY_FIELDS = [
    "group",
    "n_input",
    "n_with_hit",
    "n_no_hit",
    "n_coverage_le10",
    "coverage_le10_pct",
    "max_merged_query_coverage_pct",
]


def parse_hsp(row, lengths):
    group, name = row["group"], row["seq_name"]
    if group not in lengths or name not in lengths[group]:
        raise ValueError(f"HSP query absent from the current FASTA: {group}, {name}")
    value = {key: row[key] for key in HSP_FIELDS}
    for key in ["alignment_length", "qstart", "qend", "qlen", "query_span_bp"]:
        value[key] = int(value[key])
    for key in ["pident", "hsp_query_coverage_pct", "evalue", "bitscore"]:
        value[key] = float(value[key])
    qlen, left, right = (
        value["qlen"],
        min(value["qstart"], value["qend"]),
        max(value["qstart"], value["qend"]),
    )
    span = right - left + 1
    if (
        qlen != lengths[group][name]
        or not 1 <= left <= right <= qlen
        or value["query_span_bp"] != span
        or value["alignment_length"] < span
        or not all(
            math.isfinite(value[key])
            for key in ["pident", "hsp_query_coverage_pct", "evalue", "bitscore"]
        )
        or not 0 <= value["pident"] <= 100
        or not 0 <= value["evalue"] <= 0.5
        or value["bitscore"] < 0
        or not math.isclose(
            value["hsp_query_coverage_pct"],
            100 * span / qlen,
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
    ):
        raise ValueError(f"Invalid local alignment: {group}, {name}")
    return value


def native_rows(directory):
    for key in GENERATED:
        path = directory / f"{key}.blastn.tsv"
        if not path.with_suffix(path.suffix + ".done").is_file():
            raise ValueError(f"Completed-search marker missing: {path}")
        with path.open(newline="") as stream:
            for fields in csv.reader(stream, delimiter="\t"):
                if len(fields) != 11:
                    raise ValueError("Expected the eleven-column BLASTN output format.")
                span = abs(int(fields[5]) - int(fields[4])) + 1
                qlen = int(fields[6])
                if qlen <= 0:
                    raise ValueError("BLASTN query length must be positive.")
                yield {
                    "group": COHORTS[key],
                    "seq_name": fields[0],
                    "subject_id": fields[1],
                    "pident": fields[2],
                    "alignment_length": fields[3],
                    "qstart": fields[4],
                    "qend": fields[5],
                    "qlen": fields[6],
                    "query_span_bp": span,
                    "hsp_query_coverage_pct": 100 * span / qlen,
                    "evalue": fields[8],
                    "bitscore": fields[9],
                }


def aggregate(rows, lengths):
    alignments = {}
    for row in rows:
        hsp = parse_hsp(row, lengths)
        key = (hsp["group"], hsp["seq_name"])
        if key not in alignments:
            alignments[key] = {"intervals": [], "targets": set()}
        alignments[key]["intervals"].append(
            (min(hsp["qstart"], hsp["qend"]), max(hsp["qstart"], hsp["qend"]))
        )
        alignments[key]["targets"].add(hsp["subject_id"])
    queries, summaries = [], []
    for group, records in lengths.items():
        hits, coverage_le10, maximum_coverage = 0, 0, 0.0
        for name, qlen in records.items():
            alignment = alignments.get((group, name))
            covered_bp = 0
            if alignment is not None:
                intervals = sorted(alignment["intervals"])
                left, right = intervals[0]
                for next_left, next_right in intervals[1:]:
                    if next_left <= right + 1:
                        right = max(right, next_right)
                    else:
                        covered_bp += right - left + 1
                        left, right = next_left, next_right
                covered_bp += right - left + 1
                hits += 1
            coverage = 100 * covered_bp / qlen
            queries.append(
                {
                    "group": group,
                    "seq_name": name,
                    "length_bp": qlen,
                    "status": "hit" if alignment is not None else "no_hit",
                    "n_targets": len(alignment["targets"]) if alignment is not None else 0,
                    "n_hsps": len(alignment["intervals"]) if alignment is not None else 0,
                    "covered_bp": covered_bp,
                    "merged_query_coverage_pct": coverage,
                }
            )
            coverage_le10 += 10 * covered_bp <= qlen
            maximum_coverage = max(maximum_coverage, coverage)
        summaries.append(
            {
                "group": group,
                "n_input": len(records),
                "n_with_hit": hits,
                "n_no_hit": len(records) - hits,
                "n_coverage_le10": coverage_le10,
                "coverage_le10_pct": 100 * coverage_le10 / len(records) if records else 0.0,
                "max_merged_query_coverage_pct": maximum_coverage,
            }
        )
    return queries, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, default=TASK_DIR / "data/cohorts")
    parser.add_argument(
        "--hsps", type=Path, default=TASK_DIR / "results/blastn/core_nt.hsps.tsv"
    )
    parser.add_argument(
        "--native-dir",
        type=Path,
        default=None,
        help="Directory containing completed native <cohort>.blastn.tsv searches for all four models.",
    )
    parser.add_argument("--output-dir", type=Path, default=TASK_DIR / "results/blastn")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    targets = [args.output_dir / name for name in ["queries.tsv", "summary.tsv"]]
    ensure_available(targets, args.force)
    lengths = {
        COHORTS[key]: dna_lengths(args.sequence_dir / f"{key}.fasta")
        for key in GENERATED
    }
    if args.native_dir is not None:
        queries, summary = aggregate(native_rows(args.native_dir), lengths)
    else:
        with args.hsps.open(newline="") as stream:
            queries, summary = aggregate(
                csv.DictReader(stream, delimiter="\t"), lengths
            )
    write_table(targets[0], queries, QUERY_FIELDS)
    write_table(targets[1], summary, SUMMARY_FIELDS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
