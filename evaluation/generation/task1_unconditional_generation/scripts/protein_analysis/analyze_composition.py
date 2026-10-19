#!/usr/bin/env python3
from __future__ import annotations
import argparse
import hashlib
import subprocess
from pathlib import Path

import numpy as np
from common import ROOT, BIOLOGICAL, COHORTS, AMINO_ACIDS, fasta, protect, read_table, write_table


def union_intervals(intervals):
    merged = []
    for left, right in sorted(intervals):
        if left < 0 or right < left:
            raise ValueError("Invalid SEG interval")
        if merged and left <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
        else:
            merged.append((left, right))
    return merged


def parse_seg(text):
    intervals = {}
    name = None
    for line in text.splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
            if name in intervals:
                raise ValueError("Duplicate protein in SEG output")
            intervals[name] = []
        elif line.strip():
            if name is None:
                raise ValueError("SEG interval without a sequence header")
            intervals[name].append(tuple(map(int, line.split(" - "))))
    return {name: union_intervals(regions) for name, regions in intervals.items()}


def protein_features(protein_path, group, segmasker):
    unique = {}
    for _, sequence in fasta(protein_path):
        sequence = sequence.rstrip("*")
        if not sequence or set(sequence) - set(AMINO_ACIDS):
            raise ValueError("Expected canonical, nonempty amino-acid sequences")
        unique.setdefault(sequence, hashlib.sha256(sequence.encode("ascii")).hexdigest())
    text = "".join(f">{name}\n{sequence}\n" for sequence, name in unique.items())
    command = [str(segmasker), "-in", "-", "-outfmt", "interval", "-window", "12", "-locut", "2.2", "-hicut", "2.5"]
    masked = parse_seg(subprocess.run(command, input=text, text=True, capture_output=True, check=True).stdout)
    if set(masked) != set(unique.values()):
        raise ValueError("SEG output does not cover the distinct protein input")
    rows = []
    for sequence, name in unique.items():
        regions = masked[name]
        if any(right >= len(sequence) for _, right in regions):
            raise ValueError("SEG interval exceeds protein length")
        n_masked = sum(right - left + 1 for left, right in regions)
        rows.append({"group": group, "aa_sha256": name, "length_aa": len(sequence),
                     "seg_residues": n_masked, "seg_fraction": n_masked / len(sequence),
                     **{f"n_{aa}": sequence.count(aa) for aa in AMINO_ACIDS}})
    return rows


def summarize_features(rows):
    rows = list(rows)
    lengths = np.asarray([int(row["length_aa"]) for row in rows])
    masked = np.asarray([int(row["seg_residues"]) for row in rows])
    fractions = np.asarray([[int(row[f"n_{aa}"]) for aa in AMINO_ACIDS] for row in rows]) / lengths[:, None]
    low_complexity = masked / lengths
    group = rows[0]["group"]
    summary = {"group": group, "n_unique_proteins": len(rows), "seg_mean_pct": float(low_complexity.mean() * 100),
               "seg_zero_pct": float(np.mean(masked == 0) * 100), "seg_gt50_pct": float(np.mean(low_complexity > .5) * 100)}
    mean_fractions = fractions.mean(axis=0)
    composition = [{"group": group, "amino_acid": aa, "mean_fraction_per_unique_protein": float(mean_fractions[index])}
                   for index, aa in enumerate(AMINO_ACIDS)]
    return summary, composition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", type=Path, default=ROOT / "data/annotations")
    parser.add_argument("--scrambled-aa", type=Path, default=ROOT / "data/protein_controls/scrambled_amino_acids.faa")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/protein_composition")
    parser.add_argument("--segmasker", default="segmasker")
    parser.add_argument("--summarize-only", action="store_true", help="Use the released per-protein SEG and residue counts")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    groups = [*BIOLOGICAL, "scrambled_aa"]
    outputs = [args.output_dir / "summary.tsv", args.output_dir / "aa_composition.tsv"]
    if not args.summarize_only:
        outputs += [args.output_dir / f"{key}_protein_features.tsv.gz" for key in groups]
    protect(outputs, args.force)
    summaries, compositions = [], []
    for key in groups:
        path = args.output_dir / f"{key}_protein_features.tsv.gz"
        if args.summarize_only:
            rows = list(read_table(path))
        else:
            source = args.scrambled_aa if key == "scrambled_aa" else args.annotations_dir / key / "proteins.faa"
            display = "Scrambled-AA" if key == "scrambled_aa" else COHORTS[key]
            rows = protein_features(source, display, args.segmasker)
            write_table(path, rows, list(rows[0]))
        summary, composition = summarize_features(rows)
        summaries.append(summary)
        if key != "scrambled_aa":
            compositions.extend(composition)
    write_table(outputs[0], summaries, list(summaries[0]))
    write_table(outputs[1], compositions, list(compositions[0]))


if __name__ == "__main__":
    main()
