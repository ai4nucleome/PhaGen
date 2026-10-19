import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generation"))
from fasta_io import ROOT, header_fields, protect, read_fasta


TAG = "p2_at0.1_t1.1_tp0.9_tk4_s128_n128"
SPECIALS = ["[BOS]", "[EOS]", "[MASK]", "[PAD]", "[UNK]"]
TOKEN = re.compile("|".join(map(re.escape, SPECIALS)) + "|[ACGT]")


def count_tokens(text):
    counts = Counter()
    end = 0
    for match in TOKEN.finditer(text):
        if match.start() != end:
            raise ValueError(f"Unrecognized decoded content at character {end}")
        counts[match.group()] += 1
        end = match.end()
    if end != len(text):
        raise ValueError(f"Unrecognized decoded content at character {end}")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/sweeps/phagen_parameters_raw" / f"{TAG}.fasta",
    )
    parser.add_argument("--tag", default=TAG)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/sweeps/phagen_token_composition.tsv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "results/sweeps/phagen_token_composition.json",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    protect([args.output, args.summary], args.force)
    if (
        args.input.resolve() in {args.output.resolve(), args.summary.resolve()}
        or args.output.resolve() == args.summary.resolve()
    ):
        parser.error("Input, output and summary must be distinct files.")
    rows, total = [], Counter()
    for header, text in read_fasta(args.input):
        metadata = header_fields(header)
        counts = count_tokens(text)
        length = sum(counts.values())
        if not length or length != int(metadata["canvas_length"]):
            raise ValueError("Decoded token count differs from canvas_length.")
        total.update(counts)
        rows.append(
            {
                "tag": args.tag,
                "sequence_index": int(metadata["sequence_index"]),
                "n_tokens": length,
                **{token: counts[token] for token in SPECIALS + list("ACGT")},
                "pad_fraction": counts["[PAD]"] / length,
            }
        )
    if not rows or len({row["sequence_index"] for row in rows}) != len(rows):
        raise ValueError("Empty input or duplicate canvas indices.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    denominator = sum(total.values())
    summary = {
        "tag": args.tag,
        "canvases": len(rows),
        "total_positions": denominator,
        "token_counts": dict(total),
        "pad_fraction": total["[PAD]"] / denominator,
        "pad_percent": total["[PAD]"] / denominator * 100,
        "denominator": "all raw canvas positions including the BOS prefix; no extraction or length filtering",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
