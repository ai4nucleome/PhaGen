#!/usr/bin/env python3


from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
import subprocess

from megadna_similarity import OUTFMT, fasta_records, summarize_alignments, write_table


ROOT = Path(__file__).resolve().parent.parent
TASK_ROOT = ROOT.parent


def prepare_queries(training_fasta, output_dir, expected_records, sample_size, seed, batch_size,
                    lambda_fasta, p1_fasta):
    positions = sorted(random.Random(seed).sample(range(expected_records), sample_size))
    sampled = {position: index for index, position in enumerate(positions)}
    lengths, queries, shards = {}, [], {}
    handles = {}
    try:
        for position, (identifier, dna) in enumerate(fasta_records(training_fasta)):
            if identifier in lengths or not dna:
                raise ValueError(f"Duplicate training ID or empty sequence: {identifier}")
            lengths[identifier] = len(dna)
            if position in sampled:
                index = sampled[position]
                shard = f"background_{index // batch_size:03d}"
                if shard not in handles:
                    handles[shard] = (output_dir / "queries" / f"{shard}.fasta").open("w")
                query_id = f"background{index:04d}"
                handles[shard].write(f">{query_id}\n{dna}\n")
                metadata = {"query_id": query_id, "group": "background", "source_id": identifier,
                            "source_index": position, "length_bp": len(dna), "shard": shard}
                queries.append(metadata)
                shards.setdefault(shard, {})[query_id] = metadata
    finally:
        for handle in handles.values():
            handle.close()
    if len(lengths) != expected_records or len(queries) != sample_size:
        raise ValueError("Training FASTA or sampled-query count differs from the requested protocol")
    with (output_dir / "queries" / "controls.fasta").open("w") as handle:
        for phage, filename in (("lambda", lambda_fasta), ("p1", p1_fasta)):
            records = list(fasta_records(filename))
            if len(records) != 1 or not records[0][1]:
                raise ValueError(f"Expected one nonempty reference genome: {filename}")
            identifier, dna = records[0]
            handle.write(f">{phage}\n{dna}\n")
            metadata = {"query_id": phage, "group": "control", "source_id": identifier,
                        "source_index": -1, "length_bp": len(dna), "shard": "controls"}
            queries.append(metadata)
            shards.setdefault("controls", {})[phage] = metadata
    write_table(output_dir / "training_lengths.tsv",
                [{"source_id": name, "length_bp": length} for name, length in lengths.items()])
    write_table(output_dir / "queries.tsv", queries)
    return lengths, shards


def run_search(shard, queries, lengths, args, database):
    alignment = args.output_dir / "alignments" / f"{shard}.tsv"
    command = [args.blastn, "-query", str(args.output_dir / "queries" / f"{shard}.fasta"),
               "-db", str(database), "-task", "megablast", "-evalue", "10",
               "-max_target_seqs", "30000", "-num_threads", str(args.threads_per_job),
               "-mt_mode", "0", "-outfmt", OUTFMT, "-out", str(alignment)]
    subprocess.run(command, check=True)
    summaries, details = summarize_alignments(alignment, queries, lengths)
    if any(row["matched_subjects"] >= 30000 for row in summaries):
        raise RuntimeError("A query reached the BLAST hit cap; its score may be incomplete")
    if shard == "controls":
        for phage in ("lambda", "p1"):
            if not details[phage]:
                raise ValueError(f"No matching training sequences for {phage}")
            write_table(args.output_dir / f"{phage}_per_subject.csv", details[phage], delimiter=",")
    return summaries


def main():
    parser = argparse.ArgumentParser(description='Search lambda, P1 and sampled training genomes against an external training FASTA.')
    parser.add_argument("--training-fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for all generated files and the BLAST database.")
    parser.add_argument("--lambda-fasta", type=Path, default=TASK_ROOT / "data/lambda/NC_001416.1.fasta")
    parser.add_argument("--p1-fasta", type=Path, default=TASK_ROOT / "data/p1/NC_005856.1.fasta")
    parser.add_argument("--expected-records", type=int, default=93683)
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--threads-per-job", type=int, default=4)
    parser.add_argument("--blastn", default="blastn")
    parser.add_argument("--makeblastdb", default="makeblastdb")
    args = parser.parse_args()
    if min(args.expected_records, args.sample_size, args.batch_size, args.jobs, args.threads_per_job) < 1:
        parser.error("Record counts and parallelism must be positive")
    if args.sample_size > args.expected_records:
        parser.error("sample-size exceeds expected-records")
    args.training_fasta = args.training_fasta.resolve()
    for path in (args.training_fasta, args.lambda_fasta, args.p1_fasta):
        if not path.is_file():
            parser.error(f"FASTA not found: {path}")
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output_dir}")
    version = subprocess.check_output([args.blastn, "-version"], text=True).strip()
    args.output_dir.mkdir(parents=True)
    for name in ("queries", "alignments", "database"):
        (args.output_dir / name).mkdir()
    lengths, shards = prepare_queries(
        args.training_fasta, args.output_dir, args.expected_records, args.sample_size, args.seed,
        args.batch_size, args.lambda_fasta, args.p1_fasta,
    )
    database = args.output_dir / "database" / "training"
    subprocess.run([args.makeblastdb, "-in", str(args.training_fasta), "-dbtype", "nucl",
                    "-parse_seqids", "-out", str(database)], check=True)
    rows = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_search, shard, queries, lengths, args, database): shard
                   for shard, queries in shards.items()}
        for future in as_completed(futures):
            rows.extend(future.result())
            print(f"Finished {futures[future]}", flush=True)
    background = sorted((row for row in rows if row["group"] == "background"), key=lambda row: row["query_id"])
    controls = sorted((row for row in rows if row["group"] == "control"), key=lambda row: row["query_id"])
    if len(background) != args.sample_size or {row["query_id"] for row in controls} != {"lambda", "p1"}:
        raise ValueError("Final query count differs from the requested protocol")
    for row in controls:
        row["background_fraction_higher"] = sum(
            item["similarity_score"] > row["similarity_score"] for item in background
        ) / len(background)
    background_columns = ("query_id", "source_id", "source_index", "length_bp",
                          "matched_subjects", "similarity_score")
    control_columns = ("query_id", "source_id", "length_bp", "matched_subjects", "similarity_score",
                       "subjects_identity_gt_0_4", "subjects_identity_gt_0_9", "background_fraction_higher")
    write_table(args.output_dir / "background_scores.tsv",
                [{key: row[key] for key in background_columns} for row in background])
    write_table(args.output_dir / "control_scores.tsv",
                [{key: row[key] for key in control_columns} for row in controls])
    protocol = {"reference_records": len(lengths), "reference_bases": sum(lengths.values()),
                "sample_size": len(background), "seed": args.seed, "blastn_version": version,
                "task": "megablast", "evalue": 10, "max_target_seqs": 30000, "outfmt": OUTFMT,
                "coverage": "summed HSP alignment length divided by matched training-sequence length",
                "aggregate_coverage": "sum of subject coverages for one query",
                "overlapping_hsps": "counted separately", "self_matches": "included"}
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(args.output_dir)


if __name__ == "__main__":
    main()
