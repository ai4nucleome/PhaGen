#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from embedding_utils.encoders import MODEL_SPECS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output or args.results_dir / "model_summary.tsv"
    if output.exists() and not args.force:
        raise FileExistsError(f"Output exists: {output}; use --force")
    rows = []
    for key in MODEL_SPECS:
        row = json.loads((args.results_dir / key / "metrics.json").read_text())
        if row["model_key"] != key:
            raise ValueError(f"Unexpected model key in {key}/metrics.json")
        rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, sep="\t", index=False)


if __name__ == "__main__":
    main()
