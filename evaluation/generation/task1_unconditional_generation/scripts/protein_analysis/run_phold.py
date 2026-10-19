#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import os
import random
import subprocess
from pathlib import Path

from common import ROOT, annotation_records, fasta, protect, write_table
from summarize_phold import summarize

FIELDS = ["protein_id", "contig_id", "group", "original_gene_id", "length_aa", "partial",
          "prediction_status", "status", "phrog", "category", "product", "annotation_confidence", "prostt5_confidence"]


def write_fasta(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for name, sequence in records:
            stream.write(f">{name}\n{sequence}\n")


def run_command(command, env, log):
    with log.open("w") as stream:
        result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    return result.returncode


def predict(args, records, directory, batch_size, env):
    directory.mkdir(parents=True, exist_ok=True)
    input_path = directory / "input.faa"
    write_fasta(input_path, records)
    prediction = directory / "prediction"
    command = [args.phold, "proteins-predict", "-i", str(input_path), "-o", str(prediction),
               "-d", str(args.database_dir), "-t", str(args.predict_threads), "--batch_size", str(batch_size)]
    if run_command(command, env, directory / "prediction.log"):
        raise RuntimeError(f"Phold protein prediction failed: {directory}")
    successful = dict(fasta(prediction / "phold_3di.fasta"))
    expected = dict(records)
    failure_file = prediction / "fails.tsv"
    failures = {line.strip() for line in failure_file.read_text().splitlines() if line.strip()} if failure_file.exists() else set()
    if set(successful) & failures or set(successful) | failures != set(expected):
        raise ValueError("Phold prediction outputs do not account for every input ORF")
    if any(len(sequence) != len(expected[name]) for name, sequence in successful.items()):
        raise ValueError("Truncated 3Di prediction")
    if set(successful) - set(expected):
        raise ValueError("Unexpected protein IDs in Phold output")
    if failures and batch_size > 1:
        retried = [(name, sequence) for name, sequence in records if name in failures]
        retry_dir, retry_success = predict(args, retried, directory / "retry", 1, env)
        successful.update(retry_success)
        merged = directory / "merged_predictions"
        merged.mkdir()
        write_fasta(merged / "phold_3di.fasta", ((name, successful[name]) for name in expected if name in successful))
        amino_acids, confidence = {}, {}
        for source in (prediction, retry_dir):
            amino_acids.update(dict(fasta(source / "phold_aa.fasta")))
            for line in (source / "phold_prostT5_3di_mean_probabilities.csv").read_text().splitlines():
                name, value = line.split(",")
                confidence[name] = value
        write_fasta(merged / "phold_aa.fasta", ((name, amino_acids[name]) for name in expected if name in successful))
        (merged / "phold_prostT5_3di_mean_probabilities.csv").write_text(
            "".join(f"{name},{confidence[name]}\n" for name in expected if name in successful))
        prediction = merged
    return prediction, successful


def annotate_chunk(args, records, directory, batch_size, env):
    prediction, successful = predict(args, records, directory, batch_size, env)
    native = {}
    if successful:
        output = directory / "comparison"
        command = [args.phold, "proteins-compare", "-i", str(directory / "input.faa"), "-o", str(output),
            "--predictions_dir", str(prediction), "-d", str(args.database_dir), "-t", str(args.compare_threads), "--keep_tmp_files"]
        log = directory / "comparison.log"
        code = run_command(command, env, log)
        if code:
            raw = output / "foldseek_results.tsv"
            if not (raw.is_file() and raw.stat().st_size == 0 and "Foldseek found no hits whatsoever" in log.read_text()):
                raise RuntimeError(f"Phold structure search failed: {directory}")
        else:
            with (output / "phold_per_cds_predictions.tsv").open() as stream:
                native = {row["cds_id"]: row for row in csv.DictReader(stream, delimiter="\t")}
            if set(native) != {name for name, _ in records}:
                raise ValueError("Phold annotation IDs differ from the chunk input")
    confidence = {}
    for line in (prediction / "phold_prostT5_3di_mean_probabilities.csv").read_text().splitlines():
        name, value = line.split(",")
        confidence[name] = value
    return native, set(successful), confidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", type=Path, default=ROOT / "data/annotations")
    parser.add_argument("--cohorts-dir", type=Path, default=ROOT / "data/cohorts")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/phold")
    parser.add_argument("--database-dir", type=Path, required=True, help="External Phold 1.2.5 database including cached ProstT5 weights")
    parser.add_argument("--phold", default="phold")
    parser.add_argument("--gpu", default="0", help="One visible CUDA device for the native worker")
    parser.add_argument("--predict-threads", type=int, default=4)
    parser.add_argument("--compare-threads", type=int, default=16)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    targets = [args.output_dir / name for name in ("proteins.tsv.gz", "summary.tsv", "category_counts.tsv")]
    protect(targets, args.force)
    work = args.output_dir / "work"
    if work.exists() and any(path.is_file() and path.name != "input.faa" for path in work.rglob("*")):
        raise FileExistsError("Phold native outputs exist; select a new --output-dir")
    pairs = list(annotation_records(args.annotations_dir, args.cohorts_dir))
    metadata = {row["protein_id"]: row for row, _ in pairs}
    records = [(row["protein_id"], sequence) for row, sequence in pairs]
    if len(metadata) != len(records):
        raise ValueError("Duplicate ORF IDs")
    random.Random(20260915).shuffle(records)
    chunks = []
    for kind, pool, size, batch in (("short", [r for r in records if len(r[1]) <= 1000], 4000, 16),
                                   ("long", [r for r in records if len(r[1]) > 1000], 100, 1)):
        for index, start in enumerate(range(0, len(pool), size)):
            chunks.append((work / f"{kind}_{index:04d}", pool[start:start + size], batch))
    for directory, block, _ in chunks:
        prepared = directory / "input.faa"
        if prepared.exists() and list(fasta(prepared)) != block:
            raise ValueError("Prepared Phold chunk differs from the source proteins")
    if args.prepare_only:
        for directory, block, _ in chunks:
            write_fasta(directory / "input.faa", block)
        print(f"Prepared {len(records):,} ORFs in {len(chunks)} chunks")
        return
    if not args.database_dir.is_dir():
        raise FileNotFoundError(args.database_dir)
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_VISIBLE_DEVICES=str(args.gpu),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    version = subprocess.check_output([args.phold, "--version"], env=env, text=True)
    if "1.2.5" not in version:
        raise ValueError("This protocol requires Phold 1.2.5")
    annotations, successful, confidence = {}, set(), {}
    for directory, block, batch in chunks:
        native, predicted, scores = annotate_chunk(args, block, directory, batch, env)
        annotations.update(native)
        successful.update(predicted)
        confidence.update(scores)
    rows = []
    for gene, _ in pairs:
        name = gene["protein_id"]
        native = annotations.get(name, {})
        phrog = native.get("phrog", "")
        category = native.get("function", "unknown function")
        hit = phrog.isdigit()
        if name not in successful:
            if hit:
                raise ValueError("Failed prediction has a PHROG hit")
            status = "prediction_failed"
        elif hit:
            status = "unknown_function" if category == "unknown function" else "known_function"
        elif phrog not in {"", "NA", "nan", "None", "No_PHROG"}:
            status = "other_database_hit"
        else:
            status = "no_hit"
        rows.append({**gene, "prediction_status": "success" if name in successful else "failed", "status": status,
            "phrog": phrog, "category": category if hit else "", "product": native.get("product", ""),
            "annotation_confidence": native.get("annotation_confidence", "") if status in {"known_function", "unknown_function", "other_database_hit"} else "NA", "prostt5_confidence": confidence.get(name, "")})
    write_table(targets[0], (row for row in rows if row["group"] != "Scrambled"), FIELDS)
    summary, categories = summarize(rows)
    write_table(targets[1], summary, list(summary[0]))
    write_table(targets[2], categories, list(categories[0]))


if __name__ == "__main__":
    main()
