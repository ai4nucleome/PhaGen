#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from Bio import SeqIO


TASK_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = TASK_DIR / "data"


def load_cds(fasta_path: Path) -> tuple[str, str]:
    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    if len(records) != 1:
        raise ValueError(f"{fasta_path} should contain exactly 1 sequence, got {len(records)}.")
    rec = records[0]
    seq = str(rec.seq).upper().replace(" ", "")
    if not seq or len(seq) % 3 != 0:
        raise ValueError(f"CDS length must be a positive multiple of 3, got len={len(seq)}.")
    if any(base not in "ATCG" for base in seq):
        raise ValueError("CDS must contain only A/T/C/G bases.")
    return rec.id, seq


def read_mut_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"codon", "pos", "is_wt", "fitness_rich"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing infA design-table columns: {missing}")
    df["codon"] = df["codon"].astype(str).str.upper().str.strip()
    df["pos"] = pd.to_numeric(df["pos"], errors="raise").astype("Int64")
    df["is_wt"] = pd.to_numeric(df["is_wt"], errors="raise").astype(int)
    df["fitness_rich"] = pd.to_numeric(df["fitness_rich"], errors="raise")
    if not df["is_wt"].isin([0, 1]).all():
        raise ValueError("is_wt must contain only 0 or 1.")
    if df[["pos", "fitness_rich"]].isna().any().any():
        raise ValueError("Mutation positions and fitness_rich must not be missing.")
    return df


def replace_codon(cds: str, pos_1based: int, new_codon: str) -> tuple[str, str]:
    n_codons = len(cds) // 3
    if not (1 <= pos_1based <= n_codons):
        raise ValueError(f"pos={pos_1based} out of range (1..{n_codons}).")
    if len(new_codon) != 3 or any(base not in "ATCG" for base in new_codon):
        raise ValueError(f"invalid codon: {new_codon} (expected 3 characters from A/T/C/G).")
    start = (pos_1based - 1) * 3
    wt_codon = cds[start : start + 3]
    mut_seq = cds[:start] + new_codon + cds[start + 3 :]
    return mut_seq, wt_codon


def build_metadata(
    cds: str,
    mut_df: pd.DataFrame,
    *,
    deduplicate_wt: bool = True,
) -> pd.DataFrame:
    rows = []
    for _, row in mut_df.iterrows():
        codon = row["codon"]
        pos = int(row["pos"])
        mut_seq, wt_codon = replace_codon(cds, pos, codon)
        is_wt = int(row["is_wt"])
        if bool(is_wt) != (mut_seq == cds):
            raise ValueError(f"WT flag disagrees with the reconstructed DNA at position {pos}.")
        rows.append(
            {
                "pos": pos,
                "is_wt": is_wt,
                "mutation": f"{wt_codon}>{codon}({row.get('aa', pd.NA)})",
                "fitness": row["fitness_rich"],
                "seq": mut_seq,
            }
        )
    metadata = pd.DataFrame(rows)
    duplicate_mask = metadata.duplicated(subset="seq", keep="first")
    if duplicate_mask.any():
        duplicate_rows = metadata.loc[duplicate_mask]
        if not duplicate_rows["is_wt"].eq(1).all():
            raise ValueError("Duplicate non-WT infA sequences were generated")
        if metadata.loc[metadata["is_wt"].eq(1), "seq"].nunique() != 1:
            raise ValueError("WT-designated infA entries do not share one sequence")
        if not metadata.loc[metadata["is_wt"].eq(1), "fitness"].eq(1.0).all():
            raise ValueError("WT-designated infA entries do not all have fitness=1")
    if deduplicate_wt:
        metadata = metadata.loc[~duplicate_mask].reset_index(drop=True)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, default=DATA_DIR / "infA.fasta")
    parser.add_argument("--input_csv", type=Path, default=DATA_DIR / "infA-data.csv")
    parser.add_argument("--output_tsv", type=Path, default=DATA_DIR / "mutants_sequences.tsv")
    parser.add_argument(
        "--keep_duplicate_wt",
        action="store_true",
        help="Retain all 73 design-table WT rows for source-order cache extraction.",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing output TSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_tsv.resolve() in {args.fasta.resolve(), args.input_csv.resolve()}:
        raise ValueError("The output TSV must differ from the source files.")
    if args.output_tsv.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {args.output_tsv}; pass --force")
    _, cds = load_cds(args.fasta)
    df = read_mut_table(args.input_csv)
    meta_df = build_metadata(cds, df, deduplicate_wt=not args.keep_duplicate_wt)
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    meta_df.to_csv(args.output_tsv, sep="\t", index=False)
    print(
        f"Metadata saved to: {args.output_tsv} "
        f"({len(meta_df):,} rows, {meta_df['seq'].nunique():,} unique sequences, "
        f"{int(meta_df['is_wt'].sum())} WT row)"
    )


if __name__ == "__main__":
    main()
