#!/usr/bin/env python3
from __future__ import annotations
import argparse
import subprocess
from pathlib import Path
from common import TASK_DIR, GENERATED, dna_lengths, ensure_available


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=GENERATED, required=True)
    parser.add_argument("--sequence-dir", type=Path, default=TASK_DIR / "data/cohorts")
    parser.add_argument(
        "--database", required=True, help="Local core_nt BLAST database prefix."
    )
    parser.add_argument("--blastn-bin", default="blastn")
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output or TASK_DIR / "results/blastn" / f"{args.cohort}.blastn.tsv"
    completed = output.with_suffix(output.suffix + ".done")
    ensure_available([output, completed], args.force)
    if args.threads < 1:
        parser.error("--threads must be positive.")
    fasta = args.sequence_dir / f"{args.cohort}.fasta"
    dna_lengths(fasta)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed.unlink(missing_ok=True)
    subprocess.run(
        [
            args.blastn_bin,
            "-query",
            str(fasta),
            "-db",
            args.database,
            "-task",
            "megablast",
            "-evalue",
            "0.5",
            "-max_target_seqs",
            "100",
            "-outfmt",
            "6 qseqid sseqid pident length qstart qend qlen qcovs evalue bitscore staxid",
            "-num_threads",
            str(args.threads),
            "-out",
            str(output),
        ],
        check=True,
    )
    completed.write_text("completed\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
