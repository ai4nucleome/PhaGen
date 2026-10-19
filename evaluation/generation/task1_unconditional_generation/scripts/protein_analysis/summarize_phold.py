#!/usr/bin/env python3
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from common import ROOT, BIOLOGICAL, COHORTS, protect, read_table, write_table

CATEGORIES = ["unknown function", "DNA, RNA and nucleotide metabolism", "tail", "other", "head and packaging",
              "moron, auxiliary metabolic gene and host takeover", "lysis", "connector", "transcription regulation", "integration and excision"]
STATUSES = ["known_function", "unknown_function", "other_database_hit", "no_hit", "prediction_failed"]


def summarize(rows):
    counts, categories = Counter(), Counter()
    families = defaultdict(set)
    for row in rows:
        group = row["group"]
        if group not in [COHORTS[key] for key in BIOLOGICAL]:
            continue
        status = row["status"]
        if status not in STATUSES:
            raise ValueError(f"Unknown annotation status: {status}")
        counts[group, status] += 1
        hit = row["phrog"].isdigit()
        if hit != (status in {"known_function", "unknown_function"}):
            raise ValueError("PHROG assignment and status disagree")
        if hit:
            if row["category"] not in CATEGORIES:
                raise ValueError("Unknown PHROG functional category")
            categories[group, row["category"]] += 1
            families[group].add(row["phrog"])
    summaries, composition = [], []
    for key in BIOLOGICAL:
        group = COHORTS[key]
        n = sum(counts[group, status] for status in STATUSES)
        failed = counts[group, "prediction_failed"]
        hits = counts[group, "known_function"] + counts[group, "unknown_function"]
        if n == 0:
            raise ValueError(f"No proteins for {group}")
        summaries.append({"group": group, "n_proteins": n, "n_predicted": n - failed,
            "n_prediction_failed": failed, "n_phrog_hits": hits,
            **{f"n_{status}": counts[group, status] for status in STATUSES if status != "prediction_failed"},
            "n_distinct_phrogs": len(families[group]), "phrog_hit_percent_all_input": 100 * hits / n,
            "phrog_hit_percent_predicted": 100 * hits / (n - failed) if n > failed else "NA"})
        composition.extend({"group": group, "category": category, "n_phrog_hit_orfs": categories[group, category],
            "n_phrog_hits": hits, "fraction_of_phrog_hits": categories[group, category] / hits if hits else "NA"}
            for category in CATEGORIES)
    return summaries, composition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/phold/proteins.tsv.gz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/phold")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    targets = [args.output_dir / name for name in ("summary.tsv", "category_counts.tsv")]
    protect(targets, args.force)
    summary, categories = summarize(read_table(args.input))
    write_table(targets[0], summary, list(summary[0]))
    write_table(targets[1], categories, list(categories[0]))


if __name__ == "__main__":
    main()
