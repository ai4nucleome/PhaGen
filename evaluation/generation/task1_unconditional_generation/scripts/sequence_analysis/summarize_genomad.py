#!/usr/bin/env python3
from __future__ import annotations
import argparse
import math
import re
import statistics
from pathlib import Path
from common import (
    TASK_DIR,
    COHORTS,
    dna_lengths,
    read_table,
    write_table,
    ensure_available,
)


def summarize_cohort(cohort, fasta, directory):
    lengths = dna_lengths(fasta)
    records = read_table(directory / "classification.tsv")
    scores = {row["seq_name"]: float(row["virus_score"]) for row in records}
    if len(records) != len(scores) or set(scores) != set(lengths):
        raise ValueError(
            f"{cohort}: classification must cover all input IDs exactly once."
        )
    if any(
        not math.isfinite(value) or not 0 <= value <= 1 for value in scores.values()
    ):
        raise ValueError(f"{cohort}: invalid whole-input virus scores.")
    positive, regions = set(), set()
    for row in read_table(directory / "virus_summary.tsv"):
        name = row["seq_name"]
        if name in regions or row["fdr"] != "NA":
            raise ValueError(
                f"{cohort}: duplicate region or unexpected score calibration."
            )
        regions.add(name)
        if name in lengths:
            parent, expected_length = name, lengths[name]
        else:
            match = re.fullmatch(r"(.+)\|provirus_(\d+)_(\d+)", name)
            if match is None:
                raise ValueError(f"Unmapped viral region: {name}")
            parent, start, end = match[1], int(match[2]), int(match[3])
            if parent not in lengths or not 1 <= start <= end <= lengths[parent]:
                raise ValueError(f"Invalid provirus coordinates: {name}")
            expected_length = end - start + 1
        if (
            int(row["length"]) != expected_length
            or not 0 <= float(row["virus_score"]) <= 1
        ):
            raise ValueError(f"Invalid viral region: {name}")
        positive.add(parent)
    rows = [
        {
            "group": COHORTS[cohort],
            "seq_name": name,
            "virus_score": scores[name],
            "classified_viral": int(name in positive),
        }
        for name in lengths
    ]
    summary = {
        "group": COHORTS[cohort],
        "n_input": len(lengths),
        "positive_inputs": len(positive),
        "viral_pct": 100 * len(positive) / len(lengths),
        "mean_all_input_virus_score": statistics.mean(scores.values()),
    }
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, default=TASK_DIR / "data/cohorts")
    parser.add_argument(
        "--results-dir", type=Path, default=TASK_DIR / "results/genomad"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output_dir or args.results_dir
    targets = [output / "summary.tsv"] + [
        output / key / "per_sequence.tsv" for key in COHORTS
    ]
    ensure_available(targets, args.force)
    summaries = []
    pending = []
    for key in COHORTS:
        rows, summary = summarize_cohort(
            key, args.sequence_dir / f"{key}.fasta", args.results_dir / key
        )
        pending.append((output / key / "per_sequence.tsv", rows))
        summaries.append(summary)
    for path, rows in pending:
        write_table(path, rows)
    write_table(output / "summary.tsv", summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
