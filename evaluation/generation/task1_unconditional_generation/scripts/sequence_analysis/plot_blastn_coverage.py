#!/usr/bin/env python3
"""Plot per-sequence BLASTN union coverage, including no-hit queries at zero."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from common import COHORTS, GENERATED, TASK_DIR, ensure_available


NAME = "blastn_merged_query_coverage"
GROUPS = tuple(COHORTS[key] for key in GENERATED)
LABELS = ("PhaGen 150M", "megaDNA 145M", "Evo1-7B", "Evo2-7B")
COLORS = ("#78B77B", "#AAAAAA", "#808080", "#404040")


def load_queries(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows or {row["group"] for row in rows} != set(GROUPS):
        raise ValueError("Expected query coverage for all four generated cohorts.")
    seen = set()
    for row in rows:
        key = (row["group"], row["seq_name"])
        if not key[1] or key in seen:
            raise ValueError(f"Empty or duplicate BLAST query ID: {key}")
        seen.add(key)
        for name in ("length_bp", "n_targets", "n_hsps", "covered_bp"):
            row[name] = int(row[name])
        coverage = float(row["merged_query_coverage_pct"])
        length, covered = row["length_bp"], row["covered_bp"]
        if (
            length < 2000
            or not 0 <= covered <= length
            or not math.isfinite(coverage)
            or not 0 <= coverage <= 100
            or not math.isclose(coverage, 100 * covered / length, abs_tol=1e-10)
            or row["status"] not in ("hit", "no_hit")
            or not 0 <= row["n_targets"] <= 100
            or row["n_hsps"] < row["n_targets"]
        ):
            raise ValueError(f"Invalid merged query coverage: {key}")
        if row["status"] == "no_hit":
            valid_status = covered == row["n_targets"] == row["n_hsps"] == 0
        else:
            valid_status = covered > 0 and row["n_targets"] > 0
        if not valid_status:
            raise ValueError(f"Hit status disagrees with query coverage: {key}")
        row["merged_query_coverage_pct"] = coverage
    return rows


def build_figure(rows, seed=20261026):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 26,
            "axes.labelsize": 26,
            "xtick.labelsize": 26,
            "ytick.labelsize": 26,
            "axes.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        figure = plt.figure(figsize=(20, 10))
        axis = figure.add_axes([160 / 1440, 175 / 720, 1230 / 1440, 365 / 720])
        rng = np.random.default_rng(seed)
        for index, (group, color) in enumerate(zip(GROUPS, COLORS)):
            values = np.array(
                [row["merged_query_coverage_pct"] for row in rows if row["group"] == group]
            )
            axis.scatter(
                index + rng.uniform(-0.18, 0.18, len(values)),
                values,
                s=27,
                color=color,
                alpha=0.45,
                linewidths=0,
                rasterized=True,
                clip_on=False,
            )
        axis.set_yscale("symlog", linthresh=0.01, linscale=0.7)
        axis.set(xlim=(-0.45, 3.45), ylim=(-0.001, 210))
        axis.set_yticks(
            [0, 0.01, 0.1, 1, 10, 100], labels=["0", "0.01", "0.1", "1", "10", "100"]
        )
        axis.set_xticks(range(4), labels=LABELS)
        axis.set_ylabel("Merged query coverage (%)", labelpad=24)
        axis.tick_params(axis="x", length=0, pad=17)
        axis.tick_params(axis="y", length=6, pad=10)
        axis.grid(axis="y", linestyle="--", linewidth=0.8, color="#BBBBBB", alpha=0.5)
        axis.axhline(10, color="#527A96", linestyle="--", linewidth=1.8, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
    return figure


def render(queries, output_dir, formats=("pdf", "png"), dpi=300, seed=20261026, force=False):
    output_dir = Path(output_dir)
    targets = [output_dir / f"{NAME}.{suffix}" for suffix in formats]
    metadata_path = output_dir / f"{NAME}.metadata.json"
    ensure_available([*targets, metadata_path], force)
    rows = load_queries(queries)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure = build_figure(rows, seed)
    import matplotlib.pyplot as plt

    try:
        with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
            for path in targets:
                options = {"dpi": dpi, "bbox_inches": "tight", "pad_inches": 0.02}
                if path.suffix == ".pdf":
                    options["metadata"] = {
                        "Title": "Fig. S5B. Merged BLASTN query coverage",
                        "CreationDate": None,
                    }
                figure.savefig(path, **options)
    finally:
        plt.close(figure)
    n_low = sum(row["merged_query_coverage_pct"] <= 10 for row in rows)
    metadata = {
        "queries_sha256": hashlib.sha256(Path(queries).read_bytes()).hexdigest(),
        "coverage_definition": "Union of inclusive query intervals across all reported targets and HSPs; overlaps count once; no-hit queries have zero coverage.",
        "n_queries": len(rows),
        "n_no_hit": sum(row["status"] == "no_hit" for row in rows),
        "n_coverage_le10": n_low,
        "coverage_le10_pct": 100 * n_low / len(rows),
        "jitter_seed": seed,
        "jitter_axis": "categorical x axis only; coverage values are unchanged",
        "y_scale": {"name": "symlog", "linthresh": 0.01, "linscale": 0.7},
        "dpi": dpi,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=TASK_DIR / "results/blastn/queries.tsv")
    parser.add_argument("--output-dir", type=Path, default=TASK_DIR / "figs")
    parser.add_argument("--formats", default="pdf,png")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    formats = tuple(dict.fromkeys(args.formats.split(",")))
    if not formats or any(suffix not in ("pdf", "png", "svg") for suffix in formats) or args.dpi < 1:
        parser.error("Use pdf, png or svg formats and a positive DPI.")
    metadata = render(args.queries, args.output_dir, formats, args.dpi, args.seed, args.force)
    print(
        f"Plotted {metadata['n_queries']:,} queries; "
        f"{metadata['n_coverage_le10']:,} at or below 10% coverage "
        f"({metadata['coverage_le10_pct']:.1f}%)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
