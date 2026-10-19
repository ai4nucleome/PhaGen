#!/usr/bin/env python3
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from common import ROOT, COHORTS, fasta, protect, read_table, write_table

COMPLEMENT = str.maketrans("ACGT", "TGCA")
WINDOW_FIELDS = ["group", "contig_id", "sequence_id", "gene_id", "start", "end", "strand", "partial", "start_type", "rbs_motif", "genetic_code", "upstream_overlap_bp", "upstream_sequence"]


def upstream(sequence, start, end, strand, length=100):
    if not 1 <= start <= end <= len(sequence) or strand not in (-1, 1):
        raise ValueError("Invalid CDS coordinates")
    if strand == 1:
        left, right = start - 1 - length, start - 1
        codon = sequence[start - 1:start + 2]
    else:
        left, right = end, end + length
        codon = sequence[end - 3:end].translate(COMPLEMENT)[::-1]
    if left < 0 or right > len(sequence):
        return None, codon, (left, right)
    window = sequence[left:right]
    return window if strand == 1 else window.translate(COMPLEMENT)[::-1], codon, (left, right)


def profiles(group, counts, sizes):
    counts, sizes = np.asarray(counts), np.asarray(sizes)
    counts, sizes = counts[sizes > 0].astype(float), sizes[sizes > 0].astype(float)
    n, n_contigs = sizes.sum(), len(sizes)
    if n_contigs < 2:
        raise ValueError("Contig-cluster errors require at least two eligible contigs")
    fraction = counts.sum(axis=0) / n
    standard_error = np.sqrt(n_contigs / (n_contigs - 1) * ((counts - sizes[:, None] * fraction) ** 2).sum(axis=0)) / n
    distal = fraction[:50].mean()
    rows = [{"group": group, "position_bp": index - 100, "n_windows": int(n), "n_contigs": n_contigs,
             "ag_fraction": fraction[index], "cluster_se": standard_error[index],
             "ci95_low": max(0, fraction[index] - 1.96 * standard_error[index]),
             "ci95_high": min(1, fraction[index] + 1.96 * standard_error[index]),
             "distal_ag_fraction": distal, "ag_enrichment": fraction[index] - distal} for index in range(100)]
    summary = {"group": group, "n_windows": int(n), "n_contigs": n_contigs,
               "distal_ag_fraction": distal, "near_start_ag_fraction": fraction[80:96].mean(),
               "ag_enrichment_pp": (fraction[80:96].mean() - distal) * 100}
    return rows, summary


def calculate(annotations_dir, cohorts_dir):
    all_windows, all_profiles, summaries, eligibility = [], [], [], []
    for group_index, (key, group) in enumerate(COHORTS.items()):
        by_parent = defaultdict(list)
        genes = {}
        for row in read_table(annotations_dir / key / "genes.tsv"):
            name = row["gene"]
            start, end, strand = (int(row[column]) for column in ("start", "end", "strand"))
            if name in genes or end - start + 1 != int(row["length"]):
                raise ValueError("Invalid or duplicate gene annotation")
            genes[name] = {"gene_id": name, "start": start, "end": end, "strand": strand,
                           "rbs_motif": row["rbs_motif"], "genetic_code": row["genetic_code"]}
            by_parent[name.rsplit("_", 1)[0]].append(genes[name])
        seen = set()
        for header, _ in fasta(annotations_dir / key / "proteins.faa", full_header=True):
            parts = header.split(" # ", 4)
            name = parts[0]
            if name not in genes or name in seen:
                raise ValueError("Protein/gene annotation mismatch")
            seen.add(name)
            gene = genes[name]
            if tuple(map(int, parts[1:4])) != tuple(gene[column] for column in ("start", "end", "strand")):
                raise ValueError("Protein and gene coordinates differ")
            attributes = dict(item.split("=", 1) for item in parts[4].split(";") if "=" in item)
            gene.update({column: attributes[column] for column in ("partial", "start_type")})
        if seen != set(genes):
            raise ValueError("Protein FASTA does not cover all genes")
        counts, sizes, total = [], [], Counter()
        for parent_index, (parent, sequence) in enumerate(fasta(cohorts_dir / f"{key}.fasta")):
            if set(sequence) - set("ACGT"):
                raise ValueError("Expected ACGT-only cohort DNA")
            total["n_input_contigs"] += 1
            parent_genes = by_parent.pop(parent, [])
            coding = np.zeros(len(sequence), dtype=bool)
            for gene in parent_genes:
                if not 1 <= gene["start"] <= gene["end"] <= len(sequence):
                    raise ValueError("Gene coordinates exceed the parent sequence")
                coding[gene["start"] - 1:gene["end"]] = True
            n_windows = 0
            ag = np.zeros(100, dtype=np.int64)
            for gene in parent_genes:
                total["n_orfs"] += 1
                if gene["start_type"] == "Edge":
                    total["excluded_missing_start"] += 1
                    continue
                window, codon, (left, right) = upstream(sequence, gene["start"], gene["end"], gene["strand"])
                if codon not in {"ATG", "GTG", "TTG"} or codon != gene["start_type"]:
                    raise ValueError("The oriented start codon differs from the annotation")
                if window is None:
                    total["excluded_insufficient_upstream"] += 1
                    continue
                n_overlap = int(coding[left:right].sum())
                total["eligible_windows"] += 1
                total["upstream_overlaps_cds"] += int(n_overlap > 0)
                n_windows += 1
                bases = np.frombuffer(window.encode("ascii"), dtype=np.uint8)
                ag += (bases == ord("A")) | (bases == ord("G"))
                all_windows.append({**gene, "group": group, "contig_id": f"G{group_index}_{parent_index:06d}",
                    "sequence_id": parent, "upstream_overlap_bp": n_overlap, "upstream_sequence": window})
            counts.append(ag)
            sizes.append(n_windows)
        if by_parent:
            raise ValueError("Annotations refer to absent parent sequences")
        profile, summary = profiles(group, counts, sizes)
        all_profiles.extend(profile)
        summaries.append(summary)
        eligibility.append({"group": group, **{column: total[column] for column in
            ("n_input_contigs", "n_orfs", "eligible_windows", "excluded_missing_start", "excluded_insufficient_upstream", "upstream_overlaps_cds")}})
    return all_windows, all_profiles, summaries, eligibility


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-dir", type=Path, default=ROOT / "data/annotations")
    parser.add_argument("--cohorts-dir", type=Path, default=ROOT / "data/cohorts")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/upstream")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    names = ["upstream_windows.tsv.gz", "ag_profile.tsv", "ag_summary.tsv", "cohort_eligibility.tsv"]
    protect([args.output_dir / name for name in names], args.force)
    windows, profile, summary, eligibility = calculate(args.annotations_dir, args.cohorts_dir)
    for name, rows, fields in zip(names, [windows, profile, summary, eligibility],
                                  [WINDOW_FIELDS, list(profile[0]), list(summary[0]), list(eligibility[0])]):
        write_table(args.output_dir / name, rows, fields)


if __name__ == "__main__":
    main()
