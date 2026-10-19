#!/usr/bin/env python3
from __future__ import annotations
import csv
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[2]
COHORTS = {
    "phagen": "PhaGen",
    "megadna": "megaDNA",
    "evo1": "Evo1",
    "evo2": "Evo2",
    "natural": "Natural",
    "scrambled": "Scrambled",
}
GENERATED = tuple(list(COHORTS)[:4])


def read_fasta(path):
    name, chunks = None, []
    with Path(path).open() as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name, chunks = line[1:].split()[0], []
            else:
                if name is None:
                    raise ValueError(f"Sequence before FASTA header: {path}")
                chunks.append(line)
        if name is not None:
            yield name, "".join(chunks)


def dna_lengths(path):
    lengths = {}
    for name, sequence in read_fasta(path):
        if name in lengths or len(sequence) < 2000 or set(sequence) - set("ACGT"):
            raise ValueError(f"Duplicate ID or invalid retained sequence: {name}")
        lengths[name] = len(sequence)
    if not lengths:
        raise ValueError(f"Empty FASTA: {path}")
    return lengths


def read_table(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_table(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = iter(rows)
    first = next(rows, None)
    if fields is None:
        if first is None:
            raise ValueError("An empty table needs field names.")
        fields = list(first)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        if first is not None:
            writer.writerow(first)
        writer.writerows(rows)


def ensure_available(paths, force):
    existing = [str(path) for path in paths if Path(path).exists()]
    if existing and not force:
        raise FileExistsError(
            f"Output already exists: {existing[0]}; use --force to replace it."
        )
