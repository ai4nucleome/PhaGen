import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile

GENERATION = Path(__file__).resolve().parents[1] / "generation"
sys.path.insert(0, str(GENERATION))
from fasta_io import ROOT, protect, read_fasta


def merge_step_shards(shards, output, force=False):
    
    protect([output], force)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".extracted_spans.", suffix=".tmp", dir=output.parent
    )
    seen = set()
    try:
        with os.fdopen(descriptor, "w") as destination:
            for index, directory in enumerate(shards):
                with (directory / "extracted_spans.fasta").open() as source:
                    for line in source:
                        if line.startswith(">"):
                            parts = line[1:].rstrip("\n").split(" ", 1)
                            name = f"sh{index}_{parts[0]}"
                            if name in seen:
                                raise ValueError(
                                    f"Duplicate merged sequence ID: {name}"
                                )
                            seen.add(name)
                            rest = f" {parts[1]}" if len(parts) > 1 else ""
                            destination.write(f">{name}{rest}\n")
                        else:
                            destination.write(line)
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return len(seen)


def run_step_shards(args, row):
    
    count = int(row["num_shards"])
    samples = int(row["samples_per_shard"])
    seed_base = args.seed if args.seed is not None else int(row["seed_base"])
    stride = int(row["seed_stride"])
    batch = args.batch_size if args.batch_size is not None else int(row["batch_size"])
    dtype = args.dtype or row["dtype"]
    if min(count, samples, batch) < 1 or count * samples != int(row["num_samples"]):
        raise ValueError("Invalid recorded shard sizes or generation batch size.")
    if seed_base < 0 or stride < 1:
        raise ValueError(
            "The shard seed base must be nonnegative and its stride positive."
        )
    shards = [args.output_dir / f"shard{index}" for index in range(count)]
    merged = args.output_dir / "extracted_spans.fasta"
    targets = [merged] + [
        directory / name
        for directory in shards
        for name in ("raw_canvases.fasta", "parameters.json", "extracted_spans.fasta")
    ]
    protect(targets, args.force)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, directory in enumerate(shards):
        command = [
            sys.executable,
            str(GENERATION / "generate_phagen.py"),
            "--model",
            args.checkpoint,
            "--output-dir",
            str(directory),
            "--samples",
            str(samples),
            "--batch-size",
            str(batch),
            "--seed",
            str(seed_base + index * stride),
            "--dtype",
            dtype,
        ]
        for key, cli in [
            ("alg", "--alg"),
            ("alg_temp", "--alg-temp"),
            ("temperature", "--temperature"),
            ("top_p", "--top-p"),
            ("top_k", "--top-k"),
            ("steps", "--steps"),
            ("canvas_length", "--length"),
        ]:
            command += [cli, row[key]]
        if args.force:
            command.append("--force")
        subprocess.run(command, check=True)
        raw = directory / "raw_canvases.fasta"
        if sum(1 for _ in read_fasta(raw)) != samples:
            raise ValueError(f"Expected {samples} raw canvases in shard {index}.")
        extraction = [
            sys.executable,
            str(GENERATION / "extract_contigs.py"),
            "--model",
            "phagen",
            "--input",
            str(raw),
            "--checkpoint",
            args.checkpoint,
            "--min-length",
            row["extraction_min_length"],
            "--output",
            str(directory / "extracted_spans.fasta"),
        ]
        if args.force:
            extraction.append("--force")
        subprocess.run(extraction, check=True)
    n_spans = merge_step_shards(shards, merged, args.force)
    print(
        f"Merged {count} shards ({count * samples} canvases, {n_spans} spans) into {merged}."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        choices=["phagen_parameters", "phagen_steps", "megadna_parameters"],
        required=True,
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--source-run", help="Required for a duplicated megaDNA configuration."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-repo", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--seed",
        type=int,
        help="Override the single-run seed; for phagen_steps, override the base of the recorded shard seeds.",
    )
    parser.add_argument("--dtype", choices=["bfloat16", "float32", "float16"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config_path = (
        ROOT / "results/sweeps/megadna_parameters.tsv"
        if args.family == "megadna_parameters"
        else ROOT / "data/sweeps" / args.family / "configs.tsv"
    )
    with config_path.open() as handle:
        rows = [
            row
            for row in csv.DictReader(handle, delimiter="\t")
            if row["tag"] == args.tag
        ]
    if args.source_run:
        rows = [row for row in rows if row.get("source_run") == args.source_run]
    if len(rows) != 1:
        parser.error(
            "Choose a unique tag and source-run from the recorded configuration table."
        )
    row = rows[0]
    if args.family == "phagen_steps":
        run_step_shards(args, row)
        return
    command = [
        sys.executable,
        str(
            GENERATION
            / (
                "generate_megadna.py"
                if args.family == "megadna_parameters"
                else "generate_phagen.py"
            )
        ),
        "--model",
        args.checkpoint,
        "--output-dir",
        str(args.output_dir),
    ]
    for key, cli in [("seed", "--seed"), ("batch_size", "--batch-size")]:
        value = getattr(args, key) if getattr(args, key) is not None else row.get(key)
        if value is None:
            parser.error(
                f"The configuration table does not record {key}; supply {cli} explicitly."
            )
        command += [cli, str(value)]
    if args.family == "megadna_parameters":
        command += [
            "--single-seed",
            "--samples",
            row["n_raw"],
            "--temperature",
            row["temp"],
            "--filter-thres",
            row["filter_thres"],
        ]
        if args.model_repo:
            command += ["--model-repo", str(args.model_repo)]
    else:
        dtype = args.dtype or row.get("dtype")
        if dtype is None:
            parser.error(
                "The configuration table does not record dtype; supply --dtype explicitly."
            )
        command += ["--dtype", dtype]
        for key, cli in [
            ("alg", "--alg"),
            ("alg_temp", "--alg-temp"),
            ("temperature", "--temperature"),
            ("top_p", "--top-p"),
            ("top_k", "--top-k"),
            ("steps", "--steps"),
            ("num_samples", "--samples"),
            ("canvas_length", "--length"),
        ]:
            command += [cli, row[key]]
    if args.force:
        command.append("--force")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
