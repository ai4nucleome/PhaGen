from __future__ import annotations
import csv
import gzip
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COHORTS = {"phagen": "PhaGen", "megadna": "megaDNA", "evo1": "Evo1", "evo2": "Evo2", "natural": "Natural", "scrambled": "Scrambled"}
BIOLOGICAL = tuple(COHORTS)[:5]
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def fasta(path, full_header=False):
    name, chunks = None, []
    with Path(path).open() as stream:
        for line in stream:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                header = line[1:].rstrip()
                name, chunks = (header if full_header else header.split()[0]), []
            else:
                chunks.append(line.strip())
    if name is not None:
        yield name, "".join(chunks)


def read_table(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as stream:
        yield from csv.DictReader(stream, delimiter="\t")


def write_table(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def protect(paths, force):
    for path in map(Path, paths):
        if path.exists() and not force:
            raise FileExistsError(f"Output exists: {path}; use --force")


def annotation_records(annotations_dir, cohorts_dir):
    
    for group_index, (key, display) in enumerate(COHORTS.items()):
        parents = {name: f"G{group_index}_{index:06d}" for index, (name, _) in enumerate(fasta(cohorts_dir / f"{key}.fasta"))}
        for header, sequence in fasta(annotations_dir / key / "proteins.faa", full_header=True):
            parts = header.split(" # ", 4)
            gene = parts[0]
            parent, ordinal = gene.rsplit("_", 1)
            if parent not in parents or not ordinal.isdigit():
                raise ValueError(f"Protein does not map to a parent sequence: {gene}")
            attributes = dict(item.split("=", 1) for item in parts[4].split(";") if "=" in item)
            sequence = sequence.rstrip("*")
            if not sequence or set(sequence) - set(AMINO_ACIDS):
                raise ValueError(f"Noncanonical protein sequence: {gene}")
            contig = parents[parent]
            yield {"protein_id": f"{contig}_P{ordinal}", "contig_id": contig,
                   "group": display, "original_gene_id": gene, "length_aa": len(sequence),
                   "partial": attributes["partial"]}, sequence
