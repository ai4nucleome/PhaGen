#!/usr/bin/env python3
from __future__ import annotations
import argparse
import shutil
import subprocess
from pathlib import Path
from common import TASK_DIR, COHORTS, dna_lengths, ensure_available
from summarize_genomad import summarize_cohort
from common import write_table


def unique(directory, pattern):
    paths = list(directory.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one {pattern} in {directory}.")
    return paths[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=list(COHORTS), required=True)
    parser.add_argument("--sequence-dir", type=Path, default=TASK_DIR / "data/cohorts")
    parser.add_argument(
        "--database",
        type=Path,
        required=True,
        help="Local geNomad database version 1.9.",
    )
    parser.add_argument("--genomad-bin", default="genomad")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--splits", type=int, default=8)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--annotation-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.results_dir or TASK_DIR / "results/genomad" / args.cohort
    annotation = args.annotation_dir or TASK_DIR / "data/annotations" / args.cohort
    work = args.work_dir or TASK_DIR / "working/genomad" / args.cohort
    ensure_available(
        [
            output / name
            for name in [
                "classification.tsv",
                "virus_summary.tsv",
                "per_sequence.tsv",
                "summary.tsv",
            ]
        ]
        + [annotation / name for name in ["genes.tsv", "proteins.faa"]],
        args.force,
    )
    if args.threads < 1 or args.splits < 1:
        parser.error("--threads and --splits must be positive.")
    fasta = args.sequence_dir / f"{args.cohort}.fasta"
    dna_lengths(fasta)
    version = subprocess.check_output(
        [args.genomad_bin, "--version"], text=True
    ).strip()
    database_version = (args.database / "version.txt").read_text().strip()
    if version != "geNomad, version 1.12.0" or database_version != "1.9":
        raise ValueError("Use geNomad 1.12.0 with database 1.9.")
    subprocess.run(
        [
            args.genomad_bin,
            "end-to-end",
            "--threads",
            str(args.threads),
            "--splits",
            str(args.splits),
            str(fasta),
            str(work),
            str(args.database),
        ],
        check=True,
    )
    sources = [
        (
            unique(
                work, "*_aggregated_classification/*_aggregated_classification.json"
            ).with_suffix(".tsv"),
            output / "classification.tsv",
        ),
        (unique(work, "*_summary/*_virus_summary.tsv"), output / "virus_summary.tsv"),
        (unique(work, "*_annotate/*_genes.tsv"), annotation / "genes.tsv"),
        (unique(work, "*_annotate/*_proteins.faa"), annotation / "proteins.faa"),
    ]
    for source, target in sources:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    rows, summary = summarize_cohort(args.cohort, fasta, output)
    write_table(output / "per_sequence.tsv", rows)
    write_table(output / "summary.tsv", [summary])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
