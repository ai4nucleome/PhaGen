import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from fasta_io import ROOT, protect

ACGT = re.compile(r"[ACGT]+")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_training(args, rng):
    with args.training_statistics.open() as handle:
        metadata = list(csv.DictReader(handle, delimiter="\t"))
    training = [row for row in metadata if row["split"] == "train"]
    realm = [
        row
        for row in training
        if row["realm"] == "Duplodnaviria"
        and row["taxonomy_status"] in ("consistent", "compatible_extension")
    ]
    eligible = [
        row
        for row in realm
        if int(row["length_bp"]) >= args.min_length and int(row["non_acgt_bases"]) == 0
    ]
    unique = {}
    for row in eligible:
        unique.setdefault(row["sequence_sha256"], row)
    pool = list(unique.values())
    if len(pool) < args.per_source:
        raise ValueError("Too few unique eligible training records")
    indices = sorted(
        rng.choice(len(pool), size=args.per_source, replace=False).tolist()
    )
    selected = {pool[index]["id"]: pool[index] for index in indices}
    records = []
    parquet = pq.ParquetFile(args.train_parquet)
    if parquet.metadata.num_rows != len(training):
        raise ValueError("Training partition and statistics have different row counts")
    scanned = 0
    for batch in parquet.iter_batches(
        batch_size=args.batch, columns=["id", "source", "sequence"]
    ):
        for row in batch.to_pylist():
            identifier = row["id"]
            if identifier in selected:
                meta = selected[identifier]
                dna = row["sequence"]
                digest = hashlib.sha256(dna.encode("ascii")).hexdigest()
                if (
                    digest != meta["sequence_sha256"]
                    or len(dna) != int(meta["length_bp"])
                    or row["source"] != meta["source"]
                    or ACGT.fullmatch(dna) is None
                ):
                    raise ValueError(
                        f"Training sequence/metadata mismatch: {identifier}"
                    )
                records.append(
                    {
                        "source": "PhaGen_train",
                        "source_id": identifier,
                        "source_subset": row["source"],
                        "parquet_row": scanned,
                        "segment": "",
                        "realm": "Duplodnaviria",
                        "taxonomy": meta["historical_lineage"],
                        "taxonomy_status": meta["taxonomy_status"],
                        "length_bp": len(dna),
                        "sequence_sha256": digest,
                        "sequence": dna,
                    }
                )
            scanned += 1
    if (
        len(records) != args.per_source
        or len({r["source_id"] for r in records}) != args.per_source
    ):
        raise ValueError(
            "Selected training IDs were missing or duplicated in the source"
        )
    audit = {
        "training_records": len(training),
        "explicit_duplodnaviria": len(realm),
        "eligible_acgt_min_length": len(eligible),
        "unique_eligible_sequences": len(pool),
        "exact_duplicate_excess": len(eligible) - len(pool),
        "selected": len(records),
        "selected_source_subsets": dict(Counter(r["source_subset"] for r in records)),
    }
    print("Training selection: " + json.dumps(audit), flush=True)
    return records, audit


def sample_opengenome(args, rng, training_records):
    training_hashes = {
        bytes.fromhex(row["sequence_sha256"]) for row in training_records
    }
    seen = set()
    reservoir = []
    counts = Counter()
    parquet = pq.ParquetFile(args.opengenome_parquet)
    row_index = 0
    for batch in parquet.iter_batches(batch_size=args.batch, columns=["text"]):
        for packed in batch.column("text").to_pylist():
            if packed is None:
                counts["null_rows"] += 1
                row_index += 1
                continue
            tokens = packed.split("|")
            
            
            for offset in range(1, len(tokens) - 1, 2):
                taxonomy, dna = tokens[offset : offset + 2]
                counts["blocks_scanned"] += 1
                realms = [tag for tag in taxonomy.split(";") if tag.startswith("r__")]
                if realms != ["r__Duplodnaviria"]:
                    counts["excluded_other_missing_or_conflicting_realm"] += 1
                    continue
                counts["explicit_duplodnaviria_blocks"] += 1
                if len(dna) < args.min_length:
                    counts["excluded_below_minimum"] += 1
                    continue
                if ACGT.fullmatch(dna) is None:
                    counts["excluded_non_acgt"] += 1
                    continue
                counts["eligible_acgt_min_length_blocks"] += 1
                digest = hashlib.sha256(dna.encode("ascii")).digest()
                if digest in training_hashes:
                    counts["excluded_blocks_identical_to_selected_training"] += 1
                    continue
                if digest in seen:
                    counts["exact_duplicate_excess"] += 1
                    continue
                seen.add(digest)
                
                
                eligible_index = len(seen) - 1
                position = (
                    eligible_index
                    if eligible_index < args.per_source
                    else int(rng.integers(0, eligible_index + 1))
                )
                if position < args.per_source:
                    record = {
                        "source": "OpenGenome",
                        "source_id": f"row{row_index}_seg{(offset - 1) // 2}",
                        "source_subset": "stage2",
                        "parquet_row": row_index,
                        "segment": (offset - 1) // 2,
                        "realm": "Duplodnaviria",
                        "taxonomy": "|" + taxonomy,
                        "taxonomy_status": "explicit_source_realm",
                        "length_bp": len(dna),
                        "sequence_sha256": digest.hex(),
                        "sequence": dna,
                    }
                    if eligible_index < args.per_source:
                        reservoir.append(record)
                    else:
                        reservoir[position] = record
            row_index += 1
        if row_index % (args.batch * 1000) == 0:
            print(
                f"OpenGenome {row_index:,}/{parquet.metadata.num_rows:,} rows; "
                f"{len(seen):,} unique eligible candidates",
                flush=True,
            )
    if len(reservoir) != args.per_source:
        raise ValueError("Too few unique eligible OpenGenome records")
    reservoir.sort(key=lambda row: (row["parquet_row"], row["segment"]))
    audit = {
        "source_rows": row_index,
        **dict(counts),
        "unique_eligible_sequences_after_cross_source_exclusion": len(seen),
        "selected": len(reservoir),
    }
    print("OpenGenome selection: " + json.dumps(audit), flush=True)
    return reservoir, audit


def summarize(records):
    lengths = [row["length_bp"] for row in records]
    return {
        "n": len(records),
        "min_bp": min(lengths),
        "median_bp": statistics.median(lengths),
        "mean_bp": statistics.mean(lengths),
        "max_bp": max(lengths),
        "total_bp": sum(lengths),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--training-statistics", type=Path, required=True)
    parser.add_argument("--opengenome-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/cohorts")
    parser.add_argument("--per-source", type=int, default=5000)
    parser.add_argument("--min-length", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.per_source < 1 or args.min_length < 2000 or args.batch < 1:
        parser.error(
            "Positive sample/batch sizes and a minimum length of at least 2000 are required."
        )
    fasta, records_path, parameters = [
        args.output_dir / name
        for name in ["natural.fasta", "natural_records.tsv", "natural_sampling.json"]
    ]
    protect([fasta, records_path, parameters], args.force)
    seeds = np.random.SeedSequence(args.seed).spawn(2)
    training, training_counts = sample_training(args, np.random.default_rng(seeds[0]))
    opengenome, opengenome_counts = sample_opengenome(
        args, np.random.default_rng(seeds[1]), training
    )
    records = training + opengenome
    if len({row["sequence_sha256"] for row in records}) != len(records):
        raise ValueError("The final reference contains duplicate sequences.")
    for index, row in enumerate(records):
        row["reference_id"] = f"realdup{index:05d}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with fasta.open("w") as handle:
        for row in records:
            handle.write(
                f">{row['reference_id']} source={row['source']} source_id={row['source_id']} "
                f"parquet_row={row['parquet_row']} segment={row['segment']} realm=Duplodnaviria\n"
            )
            dna = row["sequence"]
            for start in range(0, len(dna), 80):
                handle.write(dna[start : start + 80] + "\n")
    fields = ["reference_id"] + [
        key for key in records[0] if key not in ("reference_id", "sequence")
    ]
    with records_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(records)
    parameters.write_text(
        json.dumps(
            {
                "source_counts": dict(Counter(row["source"] for row in records)),
                "seed": args.seed,
                "rng": "NumPy PCG64",
                "numpy_version": np.__version__,
                "minimum_length_bp": args.min_length,
                "summary": {
                    "combined": summarize(records),
                    "PhaGen_train": summarize(training),
                    "OpenGenome": summarize(opengenome),
                },
                "eligible_counts": {
                    "training": training_counts,
                    "opengenome": opengenome_counts,
                },
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
