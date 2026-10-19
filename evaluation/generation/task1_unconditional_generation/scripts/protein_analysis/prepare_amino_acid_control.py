#!/usr/bin/env python3
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import random
from pathlib import Path
from common import ROOT, fasta, protect, write_table


def shuffle_protein(sequence, seed=42):
    sequence_hash = hashlib.sha256(sequence.encode("ascii")).hexdigest()
    derived_seed = int(hashlib.sha256(f"{seed}:{sequence_hash}".encode("ascii")).hexdigest(), 16)
    residues = list(sequence)
    random.Random(derived_seed).shuffle(residues)
    shuffled = "".join(residues)
    if len(shuffled) != len(sequence) or Counter(shuffled) != Counter(sequence):
        raise ValueError("The control must preserve length and amino-acid counts")
    return shuffled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--natural-proteins", type=Path, default=ROOT / "data/protein_controls/natural_amino_acids.faa")
    parser.add_argument("--output", type=Path, default=ROOT / "data/protein_controls/scrambled_amino_acids.faa")
    parser.add_argument("--pairs-output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    pairs_output = args.pairs_output or args.output.with_name("paired_proteins.tsv")
    protect([args.output, pairs_output], args.force)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pairs = []
    with args.output.open("w") as stream:
        for name, sequence in fasta(args.natural_proteins):
            shuffled_name = name + "_scrambled"
            stream.write(f">{shuffled_name}\n{shuffle_protein(sequence)}\n")
            pairs.append({"natural_protein_id": name, "scrambled_protein_id": shuffled_name, "length_aa": len(sequence)})
    write_table(pairs_output, pairs, list(pairs[0]))


if __name__ == "__main__":
    main()
